#!/usr/bin/env python3
"""
Basic test script for Actor framework without requiring full OmniGen2 model.
This tests the Ray setup, configuration, and basic data structures.
"""

import os
import sys
import torch
import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf
import logging
from unittest.mock import Mock, patch

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verl import DataProto
from tensordict import TensorDict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_minimal_config():
    """Create a minimal test configuration"""
    config = OmegaConf.create({
        'model': {
            'path': 'OmniGen2/OmniGen2',
            'trust_remote_code': True,
            'enable_gradient_checkpointing': False,
            'use_liger': False,
            'cfg_weight': 1.0,
            'use_remove_padding': False,
            'override_config': {}
        },
        'actor': {
            'fsdp_config': {
                'fsdp_size': -1,
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
                'total_training_steps': 100,
                'lr_warmup_steps_ratio': 0.1
            },
            'ppo_mini_batch_size': 4,
            'ppo_micro_batch_size': 2,
            'ppo_epochs': 1,
            'ulysses_sequence_parallel_size': 1,
            'checkpoint': {
                'contents': ['model', 'optimizer', 'lr_scheduler']
            }
        },
        'rollout': {
            'name': 'hf',
            'n': 1,
            'temperature': 1.0,
            'response_length': 64,
            'log_prob_micro_batch_size': 2,
            'log_prob_max_token_len_per_gpu': 1024,
            'log_prob_use_dynamic_bsz': False,
            'tensor_model_parallel_size': 1,
            'cfg_weight': 1.0
        }
    })
    return config

def create_test_data_proto(batch_size=2, seq_len=64):
    """Create test DataProto with realistic structure for PPO"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    prompt_len = seq_len // 2
    response_len = seq_len - prompt_len
    
    batch_data = TensorDict({
        # Input sequences
        'input_ids': torch.randint(1, 1000, (batch_size, seq_len), dtype=torch.long).to(device),
        'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long).to(device),
        'position_ids': torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(device),
        
        # PPO specific data
        'prompts': torch.randint(1, 1000, (batch_size, prompt_len), dtype=torch.long).to(device),
        'responses': torch.randint(1, 1000, (batch_size, response_len), dtype=torch.long).to(device),
        'old_log_probs': torch.randn(batch_size, response_len).to(device),
        'values': torch.randn(batch_size, response_len).to(device),
        'rewards': torch.randn(batch_size, response_len).to(device),
        'advantages': torch.randn(batch_size, response_len).to(device),
        
        # Image data for OmniGen2
        'gen_img': torch.randint(0, 255, (batch_size, 512, 512, 3), dtype=torch.uint8).to(device),
        'seq_img_mask': torch.zeros(batch_size, seq_len, dtype=torch.long).to(device),
    }, batch_size=batch_size)
    
    # Set some response positions as image tokens
    batch_data['seq_img_mask'][:, -response_len:] = torch.randint(0, 2, (batch_size, response_len))
    
    meta_info = {
        'global_token_num': batch_size * seq_len,
        'eos_token_id': 2,
        'pad_token_id': 0,
        'temperature': 1.0,
        'micro_batch_size': 1,
        'max_token_len': seq_len,
        'use_dynamic_bsz': False
    }
    
    return DataProto(batch=batch_data, meta_info=meta_info)

@ray.remote(num_cpus=1)
class MockActorTester:
    """Mock actor for testing framework functionality"""
    
    def __init__(self, config):
        self.config = config
        self.initialized = False
        
    def test_config_structure(self):
        """Test configuration structure"""
        required_keys = ['model', 'actor', 'rollout']
        missing_keys = [key for key in required_keys if key not in self.config]
        
        if missing_keys:
            return False, f"Missing config keys: {missing_keys}"
        
        return True, "Configuration structure is valid"
    
    def test_data_processing(self, data_proto):
        """Test data processing capabilities"""
        try:
            batch_size = data_proto.batch.batch_size[0]
            
            # Test data shapes
            input_shape = data_proto.batch['input_ids'].shape
            img_shape = data_proto.batch['gen_img'].shape
            
            # Test chunking
            chunks = data_proto.chunk(2)
            
            # Test device movement
            data_cpu = data_proto.to('cpu')
            if torch.cuda.is_available():
                data_cuda = data_proto.to('cuda')
            
            info = {
                'batch_size': batch_size,
                'input_shape': input_shape,
                'img_shape': img_shape,
                'num_chunks': len(chunks),
                'meta_info_keys': list(data_proto.meta_info.keys())
            }
            
            return True, info
            
        except Exception as e:
            return False, f"Data processing failed: {str(e)}"
    
    def simulate_training_step(self, data_proto):
        """Simulate a training step"""
        try:
            batch_size = data_proto.batch.batch_size[0]
            response_len = data_proto.batch['responses'].shape[1]
            
            # Simulate PPO computations
            advantages = data_proto.batch['advantages']
            old_log_probs = data_proto.batch['old_log_probs']
            values = data_proto.batch['values']
            
            # Mock metrics
            metrics = {
                'actor/policy_loss': torch.randn(1).item(),
                'actor/value_loss': torch.randn(1).item(),
                'actor/entropy': torch.randn(1).item(),
                'actor/approx_kl': torch.abs(torch.randn(1)).item(),
                'actor/clipfrac': torch.rand(1).item(),
                'perf/mfu/actor': torch.rand(1).item() * 0.5,
                'perf/max_memory_allocated_gb': 2.5,
                'actor/lr': 1e-5
            }
            
            # Create mock result
            result_meta = {'metrics': metrics}
            result = DataProto(meta_info=result_meta)
            
            return True, result
            
        except Exception as e:
            return False, f"Training step simulation failed: {str(e)}"

def test_ray_functionality():
    """Test Ray setup and remote execution"""
    print("\n🚀 Testing Ray functionality...")
    
    try:
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                num_cpus=2,
                num_gpus=1 if torch.cuda.is_available() else 0,
                log_to_driver=False
            )
        
        print("✅ Ray initialized successfully")
        
        # Test remote function
        @ray.remote
        def simple_task(x):
            return x * 2
        
        result = ray.get(simple_task.remote(21))
        assert result == 42, f"Expected 42, got {result}"
        print("✅ Ray remote execution works")
        
        return True
    
    except Exception as e:
        print(f"❌ Ray test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🔬 Starting Basic Actor Framework Test...")
    
    # Set up distributed environment variables for testing (for consistency)
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('LOCAL_RANK', '0')
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12345')
    
    # Test 1: Ray functionality
    if not test_ray_functionality():
        return
    
    # Test 2: Configuration and data structures
    print("\n📋 Testing configuration and data structures...")
    config = create_minimal_config()
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
        print(f"📁 Mock config using local path: {model_path}")
    else:
        print(f"📡 Mock config using HuggingFace identifier: {model_path}")
    
    print("✅ Configuration created")
    
    data_proto = create_test_data_proto(batch_size=2, seq_len=64)
    print("✅ DataProto created")
    
    # Test 3: Mock actor testing
    print("\n🎭 Testing mock actor functionality...")
    try:
        mock_actor = MockActorTester.remote(config)
        
        # Test config structure
        success, message = ray.get(mock_actor.test_config_structure.remote())
        if success:
            print(f"✅ Config test: {message}")
        else:
            print(f"❌ Config test failed: {message}")
            return
        
        # Test data processing
        success, result = ray.get(mock_actor.test_data_processing.remote(data_proto))
        if success:
            print(f"✅ Data processing test passed")
            print(f"   - Batch size: {result['batch_size']}")
            print(f"   - Input shape: {result['input_shape']}")
            print(f"   - Image shape: {result['img_shape']}")
            print(f"   - Chunks created: {result['num_chunks']}")
        else:
            print(f"❌ Data processing test failed: {result}")
            return
        
        # Test training step simulation
        success, result = ray.get(mock_actor.simulate_training_step.remote(data_proto))
        if success:
            print("✅ Training step simulation passed")
            metrics = result.meta_info['metrics']
            print(f"   - Policy loss: {metrics['actor/policy_loss']:.4f}")
            print(f"   - Value loss: {metrics['actor/value_loss']:.4f}")
            print(f"   - Entropy: {metrics['actor/entropy']:.4f}")
            print(f"   - Approx KL: {metrics['actor/approx_kl']:.4f}")
        else:
            print(f"❌ Training step simulation failed: {result}")
            return
        
        print("\n🎉 All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Mock actor test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if ray.is_initialized():
            ray.shutdown()
            print("🧹 Ray shutdown complete")

def test_torch_and_dependencies():
    """Test PyTorch and key dependencies"""
    print("\n🔧 Testing PyTorch and dependencies...")
    
    try:
        # Test PyTorch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"   - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   - CUDA devices: {torch.cuda.device_count()}")
            print(f"   - Current device: {torch.cuda.current_device()}")
        
        # Test tensor operations
        x = torch.randn(2, 3)
        y = x.cuda() if torch.cuda.is_available() else x
        z = y * 2
        print("✅ Basic tensor operations work")
        
        # Test TensorDict
        td = TensorDict({'a': torch.randn(2, 3), 'b': torch.randn(2, 4)}, batch_size=2)
        print("✅ TensorDict works")
        
        # Test OmegaConf
        cfg = OmegaConf.create({'test': 'value'})
        print("✅ OmegaConf works")
        
        return True
        
    except Exception as e:
        print(f"❌ Dependency test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Basic Actor Framework Test")
    print("=" * 60)
    
    # Test dependencies first
    if not test_torch_and_dependencies():
        print("❌ Dependency tests failed, exiting...")
        sys.exit(1)
    
    # Run main tests
    main()
    
    print("\n" + "=" * 60)
    print("🏁 Basic test completed!")
    print("=" * 60)
    print("\n💡 To test with a real OmniGen2 model, use test_actor_omnigen.py")
    print("   and update the model path in the configuration.") 