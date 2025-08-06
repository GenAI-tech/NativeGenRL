#!/usr/bin/env python3
"""
Test script for OmniGen2 Actor in Ray and Verl framework.
This script tests the actor functionality individually.
"""

import os
import sys
import torch
import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf
import logging

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from tensordict import TensorDict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_config():
    """Create a test configuration for OmniGen2 actor"""
    config = OmegaConf.create({
        'model': {
            'path': 'OmniGen2/OmniGen2',  # Replace with actual OmniGen2 model path
            'trust_remote_code': True,
            'enable_gradient_checkpointing': False,
            'use_liger': False,
            'cfg_weight': 1.0,
            'use_remove_padding': False,
            'override_config': {}
        },
        'actor': {
            'fsdp_config': {
                'fsdp_size': -1,  # Use all available GPUs
                'param_offload': False,
                'optimizer_offload': False,
                'mixed_precision': {
                    'param_dtype': 'bf16',
                    'reduce_dtype': 'fp32', 
                    'buffer_dtype': 'fp32'
                },
                'wrap_policy': None
            },
            'optim': {
                'lr': 1e-5,
                'betas': [0.9, 0.999],
                'weight_decay': 0.01,
                'total_training_steps': 1000,
                'lr_warmup_steps_ratio': 0.1
            },
            'ppo_mini_batch_size': 4,  # Will be normalized by world size
            'ppo_micro_batch_size': 2,  # Will be normalized by world size  
            'ppo_epochs': 1,
            'ulysses_sequence_parallel_size': 1,
            'checkpoint': {'contents': ['model', 'hf_model', 'optimizer', 'extra']},
        },
        'rollout': {
            'name': 'hf',
            'n': 1,
            'temperature': 1.0,
            'response_length': 128,
            'log_prob_micro_batch_size': 2,
            'log_prob_max_token_len_per_gpu': 2048,
            'log_prob_use_dynamic_bsz': False,
            'tensor_model_parallel_size': 1,
            'cfg_weight': 1.0
        }
    })
    return config

def create_dummy_data_proto(batch_size=2, seq_len=128, vocab_size=32000):
    """Create dummy DataProto for testing"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy batch data
    batch_data = TensorDict({
        'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long).to(device),
        'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long).to(device),
        'position_ids': torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(device),
        'prompts': torch.randint(0, vocab_size, (batch_size, seq_len//2), dtype=torch.long).to(device),
        'responses': torch.randint(0, vocab_size, (batch_size, seq_len//2), dtype=torch.long).to(device),
        'old_log_probs': torch.randn(batch_size, seq_len//2).to(device),
        'values': torch.randn(batch_size, seq_len//2).to(device),
        'rewards': torch.randn(batch_size, seq_len//2).to(device),
        'advantages': torch.randn(batch_size, seq_len//2).to(device),
        'gen_img': torch.randint(0, 255, (batch_size, 1024, 1024, 3), dtype=torch.uint8).to(device),
        'seq_img_mask': torch.zeros(batch_size, seq_len, dtype=torch.long).to(device)
    }, batch_size=batch_size)
    
    # Create meta info
    meta_info = {
        'global_token_num': batch_size * seq_len,
        'eos_token_id': 151643,
        'pad_token_id': 151643,
        'temperature': 1.0,
        'micro_batch_size': 2,
        'max_token_len': seq_len,
        'use_dynamic_bsz': False
    }
    
    return DataProto(batch=batch_data, meta_info=meta_info)

@ray.remote(num_gpus=1)
class TestActorWorker:
    """Ray remote worker for testing actor"""
    
    def __init__(self, config):
        self.actor_worker = ActorRolloutRefWorker(config, role='actor')
        
    def init_model(self):
        """Initialize the actor model"""
        try:
            self.actor_worker.init_model()
            return True, "Model initialized successfully"
        except Exception as e:
            return False, f"Model initialization failed: {str(e)}"
    
    def test_update_actor(self, data_proto):
        """Test actor update"""
        try:
            result = self.actor_worker.update_actor(data_proto)
            return True, f"Actor update successful, metrics: {result.meta_info.get('metrics', {})}"
        except Exception as e:
            return False, f"Actor update failed: {str(e)}"
    
    def get_actor_info(self):
        """Get actor information"""
        try:
            info = {
                'has_actor': hasattr(self.actor_worker, 'actor'),
                'has_tokenizer': hasattr(self.actor_worker, 'tokenizer'),
                'has_processor': hasattr(self.actor_worker, 'processor'),
                'device_count': torch.cuda.device_count(),
                'world_size': self.actor_worker.world_size if hasattr(self.actor_worker, 'world_size') else 1
            }
            return True, info
        except Exception as e:
            return False, f"Failed to get actor info: {str(e)}"

def main():
    """Main test function"""
    print("Starting OmniGen2 Actor Test...")
    
    # Set up distributed environment variables for testing
    # These are required by the Actor base class which expects a distributed PyTorch environment
    os.environ.setdefault('WORLD_SIZE', '1')      # Total number of processes (1 for single-node testing)
    os.environ.setdefault('RANK', '0')            # Global rank of this process (0 for master)
    os.environ.setdefault('LOCAL_RANK', '0')      # Local rank within the node (0 for single GPU)
    os.environ.setdefault('MASTER_ADDR', 'localhost')  # Address of the master node
    os.environ.setdefault('MASTER_PORT', '12345')      # Port for distributed communication
    
    # Check if OmniGen2 model path exists (for local paths) or is a valid HF identifier
    config = create_test_config()
    model_path = config.model.path
    
    # Check if it's a local path vs HuggingFace identifier
    is_local_path = (
        model_path.startswith('/') or 
        model_path.startswith('./') or 
        model_path.startswith('../') or 
        '\\' in model_path or
        (len(model_path.split('/')) > 2)
    )
    
    if is_local_path:
        # It's a local path, check if it exists
        if not os.path.exists(model_path):
            print(f"⚠️  Local model path {model_path} does not exist!")
            print("Please update the model path in create_test_config() function")
            print("You can use either:")
            print("  - A local path: '/path/to/omnigen2/model'")
            print("  - A HuggingFace identifier: 'OmniGen2/OmniGen2'")
            return
        else:
            print(f"📁 Using local model: {model_path}")
    else:
        # Assume it's a HuggingFace identifier
        print(f"📡 Using HuggingFace model: {model_path}")
        print("   Model will be downloaded automatically if not cached")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=4,
            num_gpus=1 if torch.cuda.is_available() else 0
        )
    
    try:
        print("🚀 Initializing Ray worker...")
        test_worker = TestActorWorker.remote(config)
        
        # Test 1: Get actor info
        print("\n📊 Testing actor info...")
        success, info = ray.get(test_worker.get_actor_info.remote())
        if success:
            print(f"✅ Actor info: {info}")
        else:
            print(f"❌ Failed to get actor info: {info}")
            return
        
        # Test 2: Initialize model
        print("\n🔧 Testing model initialization...")
        success, message = ray.get(test_worker.init_model.remote())
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
            return
        
        # Test 3: Create dummy data and test actor update
        print("\n🎯 Testing actor update...")
        dummy_data = create_dummy_data_proto(batch_size=2, seq_len=128)
        print(f"Created dummy data with batch size: {dummy_data.batch.batch_size}")
        
        success, message = ray.get(test_worker.test_update_actor.remote(dummy_data))
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up Ray
        ray.shutdown()
        print("🧹 Ray shutdown complete")

def test_data_proto():
    """Test DataProto creation and manipulation"""
    print("\n🧪 Testing DataProto creation...")
    
    try:
        data = create_dummy_data_proto(batch_size=2, seq_len=64)
        print(f"✅ Created DataProto with batch_size: {data.batch.batch_size}")
        print(f"   - input_ids shape: {data.batch['input_ids'].shape}")
        print(f"   - gen_img shape: {data.batch['gen_img'].shape}")
        print(f"   - meta_info keys: {list(data.meta_info.keys())}")
        
        # Test chunking
        chunks = data.chunk(2)
        print(f"✅ Successfully chunked data into {len(chunks)} chunks")
        
        return True
    except Exception as e:
        print(f"❌ DataProto test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 OmniGen2 Actor Test Script")
    print("=" * 60)
    
    # Test DataProto first
    if not test_data_proto():
        print("❌ DataProto test failed, exiting...")
        sys.exit(1)
    
    # Test actor
    main()
    
    print("\n" + "=" * 60)
    print("🏁 Test script completed!")
    print("=" * 60) 