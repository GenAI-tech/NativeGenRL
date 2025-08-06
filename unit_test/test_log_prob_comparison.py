#!/usr/bin/env python3
"""
Test script to compare log probabilities from two methods:
1. Pipeline.__call__() - generates images and returns log probs for all steps
2. Pipeline.compute_log_prob() - computes log prob for specific stored transitions

This test verifies that both methods produce the same log probability values.
"""

import os
import sys
import torch
import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf
import logging
from datetime import datetime
import PIL.Image
import random

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from tensordict import TensorDict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_random_seed(seed=42):
    """Set fixed random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # print(f"🎲 Set random seed to {seed} for reproducibility")

def create_test_config():
    """Create a configuration for OmniGen2 testing"""
    config = OmegaConf.create({
        'use_chat_pipeline': False,
        'model': {
            'path': 'OmniGen2/OmniGen2',  # HuggingFace model identifier
            'trust_remote_code': True,
            'enable_gradient_checkpointing': True,
            'use_liger': False,
            'cfg_weight': 1.0,  # Guidance scale for better image quality
            'use_remove_padding': False,
            'detach_uncond': True,
            'override_config': {}
        },
        'actor': {
            'ppo_mini_batch_size': 1,
            'ppo_micro_batch_size': None,
            'ppo_micro_batch_size_per_gpu': 1,
            'use_kl_loss': False,
            'kl_loss_coef': 0.0,
            'kl_loss_type': 'low_var_kl',
            'entropy_coeff': -0.00,
            'ignore_img_start': False,
            'update_mode': 'image',
            'checkpoint': {'contents': ['model', 'hf_model', 'optimizer', 'extra']},
            'optim': {
                'lr': 5e-6
            },
            'adaptive_entropy_coeff': {
                'enable': False,
                'text': {
                    'init_alpha': 0.0,
                    'target_entropy': 2.0,
                    'lr': 5e-5,
                    'min_coeff': -5e-3,
                    'max_coeff': 5e-3
                }
            },
            'fsdp_config': {
                'fsdp_size': -1,
                'param_offload': False,  # Disable param offload to avoid device issues
                'optimizer_offload': False,  # Disable optimizer offload
                'mixed_precision': {
                    'param_dtype': 'bf16',
                    'reduce_dtype': 'fp32', 
                    'buffer_dtype': 'fp32'
                },
                'wrap_policy': {
                    'min_num_params': 100000000
                }
            },
            'ulysses_sequence_parallel_size': 1,
            'noise_level': 0.3,  # Add noise_level for diffusion
            'num_inference_steps': 5  # Reduced from 10 to match rolling and save memory
        },
        'rollout': {
            'name': 'hf',  # Use HuggingFace rollout for OmniGen2
            'n': 1, # Number of images to generate per prompt
            'temperature': 1.0,
            'response_length': 128,  # Token response length
            'log_prob_micro_batch_size': None,
            'log_prob_micro_batch_size_per_gpu': 4,
            'log_prob_max_token_len_per_gpu': 2048,
            'log_prob_use_dynamic_bsz': False,
            'tensor_model_parallel_size': 1,
            'gpu_memory_utilization': 0.6,
            'micro_batch_size': 16,
            'cot_generate': True,
            'val_kwargs': {
                'do_sample': True
            },
            'text_guidance_scale': 5.0,
            'image_guidance_scale': 2.0,
            'height': 256,  # Reduced from 384 to save memory
            'width': 256,   # Reduced from 384 to save memory  
            'num_inference_steps': 5,  # Reduced from 10 to save memory
            'scheduler': 'euler'  # Options: 'euler', 'dpmsolver++'
        },
        'ref': {
            'log_prob_micro_batch_size_per_gpu': 4,
            'num_inference_steps': 5,  # Match actor config and reduced for memory
            'fsdp_config': {
                'param_offload': True
            }
        },
        'algorithm': {
            'kl_ctrl': {
                'kl_coef': 0.000
            },
            'filter_groups': {
                'enable': False,
                'max_num_gen_batches': 16
            }
        },
        'trainer': {
            'critic_warmup': 0
        }
    })
    return config

def create_prompt_data_proto(prompts, tokenizer):
    """Create DataProto from text prompts"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = len(prompts)
    
    # Tokenize prompts
    tokenized = tokenizer(
        prompts, 
        return_tensors='pt', 
        padding="longest", 
        truncation=True, 
        max_length=512
    )
    
    # Move tensors to device
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    seq_len = input_ids.shape[1]
    
    # Create position ids
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Create batch data
    batch_data = TensorDict({
        'input_ids': input_ids,
        'attention_mask': attention_mask, 
        'position_ids': position_ids,
    }, batch_size=batch_size)
    
    # Create meta info
    meta_info = {
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'image_start_token_id': tokenizer.pad_token_id,
        'do_sample': True,
        'response_length': 128,
        'temperature': 1.0,
        'device': device
    }
    
    # Add raw prompt ids for non_tensor_batch
    raw_prompt_ids = [
        tokenizer.encode(prompt, add_special_tokens=False) 
        for prompt in prompts
    ]
    non_tensor_data = {
        'raw_prompt_ids': np.array(raw_prompt_ids, dtype=object)
    }
    
    return DataProto(batch=batch_data, meta_info=meta_info, non_tensor_batch=non_tensor_data)

@ray.remote(num_gpus=1)
class LogProbTestWorker:
    """Ray remote worker for testing log probability comparison"""
    
    def __init__(self, config):
        self.config = config
        self.worker = None
        
    def init_model(self):
        """Initialize the model"""
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                device = torch.cuda.current_device()
                print(f"Using CUDA device: {device}")
            
            self.worker = ActorRolloutRefWorker(self.config, role='actor_rollout')
            self.worker.init_model()
            
            print("✅ Model initialized successfully!")
            return True, "Model initialized successfully"
        except Exception as e:
            import traceback
            return False, f"Model initialization failed: {str(e)}\n{traceback.format_exc()}"
    
    def get_tokenizer(self):
        """Get the tokenizer"""
        try:
            if self.worker and hasattr(self.worker, 'tokenizer'):
                return True, self.worker.tokenizer
            else:
                return False, "Tokenizer not available"
        except Exception as e:
            return False, f"Failed to get tokenizer: {str(e)}"
    
    def test_log_prob_comparison(self, prompt_data):
        """Test comparing log probs from generation vs recomputation"""
        try:
            if self.worker is None:
                return False, "Worker not initialized", None, None
                
            device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
            prompt_data = prompt_data.to(device)
            
            # Set fixed seed before Method 1
            set_random_seed(42)
            
            print("🔄 Method 1: Generating images with stored diffusion data...")
            
            # Generate images using the worker (this should return diffusion_data)
            result = self.worker.generate_sequences(prompt_data)
            
            print(f"✅ Method 1 completed")
            print(f"   Result batch keys: {list(result.batch.keys())}")
            if hasattr(result, 'non_tensor_batch'):
                print(f"   Result non_tensor keys: {list(result.non_tensor_batch.keys())}")
            
            # Check if we have the required data for Method 2
            required_keys = ['latents', 'next_latents', 'timesteps', 'log_probs']
            missing_keys = [key for key in required_keys if key not in result.batch]
            
            if missing_keys:
                return False, f"Missing required keys for Method 2: {missing_keys}", result, None
            
            # Extract data from Method 1
            log_probs_method1 = result.batch['log_probs']  # (batch_size, num_steps)
            
            print(f"   Log probs shape: {log_probs_method1.shape}")
            print(f"   Log probs range: {log_probs_method1.min().item():.4f} to {log_probs_method1.max().item():.4f}")
            
            # Move Method 1 results to CPU to free GPU memory
            log_probs_method1 = log_probs_method1.cpu()
            
            # Clear GPU cache to free memory before Method 2
            torch.cuda.empty_cache()
            print("   Cleared GPU cache to free memory")
            
            # Set the same fixed seed before Method 2
            set_random_seed(42)
            
            print("🔄 Method 2: Recomputing log probs using compute_log_prob...")
            
            # Extract diffusion data for Method 2
            diffusion_data = {
                'latents': result.batch['latents'],
                'next_latents': result.batch['next_latents'],
                'timesteps': result.batch['timesteps'],
                # 'prompt_embeds': result.batch['prompt_embeds'],
                # 'prompt_attention_mask': result.batch['prompt_attention_mask'],
                'negative_prompt_embeds': result.batch['negative_prompt_embeds'],
                'negative_prompt_attention_mask': result.batch['negative_prompt_attention_mask'],
            }
            
            # Helper to robustly convert various Python/numpy objects to torch tensors (or lists of tensors)
            def _convert_to_tensor(obj, device, dtype):
                """Recursively convert obj to torch.Tensor (or list of tensors) on given device/dtype."""
                if torch.is_tensor(obj):
                    return obj.to(device=device, dtype=dtype)
                elif isinstance(obj, np.ndarray):
                    if obj.dtype == object:
                        converted = [_convert_to_tensor(o, device, dtype) for o in obj]
                        # Try stacking if shapes match
                        try:
                            return torch.stack(converted, dim=0)
                        except Exception:
                            return converted
                    else:
                        return torch.from_numpy(obj).to(device=device, dtype=dtype)
                elif isinstance(obj, (list, tuple)):
                    converted = [_convert_to_tensor(o, device, dtype) for o in obj]
                    try:
                        return torch.stack(converted, dim=0)
                    except Exception:
                        return converted
                else:
                    # Scalar or unsupported type, try direct tensor conversion
                    try:
                        return torch.tensor(obj, device=device, dtype=dtype)
                    except Exception:
                        return obj  # Fallback: return as-is if still unsupported

            # Add optional keys including noise if available
            for key in ['ref_latents', 'freqs_cis', 'noise', 'noise_pred', 'prompt_embeds', 'prompt_attention_mask']:
                if key in result.batch:
                    diffusion_data[key] = result.batch[key]
                elif hasattr(result, 'non_tensor_batch') and key in result.non_tensor_batch:
                    non_tensor_data = result.non_tensor_batch[key]
                    if key == 'prompt_embeds':
                        converted = _convert_to_tensor(non_tensor_data, 
                                                    device=diffusion_data['negative_prompt_embeds'].device,
                                                    dtype=diffusion_data['negative_prompt_embeds'].dtype)
                    elif key == 'prompt_attention_mask':  # dtype different from prompt embeds
                        converted = _convert_to_tensor(non_tensor_data, 
                                                    device=diffusion_data['negative_prompt_attention_mask'].device,
                                                    dtype=diffusion_data['negative_prompt_attention_mask'].dtype)
                    else:
                        converted = _convert_to_tensor(non_tensor_data, 
                                                    device=diffusion_data['negative_prompt_embeds'].device,
                                                    dtype=diffusion_data['negative_prompt_embeds'].dtype)
                    diffusion_data[key] = converted
            print(f"   Diffusion data keys: {list(diffusion_data.keys())}")
            
            # Ensure all tensors are on the same device
            for key, value in diffusion_data.items():
                if torch.is_tensor(value):
                    diffusion_data[key] = value.to(device)
                    print(f"   Moved {key} to device: {device}")
            
            # Use the pipeline's compute_log_prob method directly
            pipeline = self.worker.actor_module_fsdp
            
            # Set model to eval mode for consistency
            # pipeline.eval()
            
            with torch.no_grad():
                log_probs_method2 = pipeline.compute_log_prob(
                    latents=diffusion_data['latents'],
                    next_latents=diffusion_data['next_latents'],
                    timesteps=diffusion_data['timesteps'],
                    prompt_embeds=diffusion_data['prompt_embeds'],
                    prompt_attention_mask=diffusion_data['prompt_attention_mask'],
                    negative_prompt_embeds=diffusion_data['negative_prompt_embeds'],
                    negative_prompt_attention_mask=diffusion_data['negative_prompt_attention_mask'],
                    ref_latents=diffusion_data.get('ref_latents', None),
                    noise_level=self.config.actor.get('noise_level', 0.3),
                    num_inference_steps=self.config.actor.get('num_inference_steps', 5),  # Updated to match config
                    text_guidance_scale=self.config.rollout.get('text_guidance_scale', 5.0),
                    image_guidance_scale=self.config.rollout.get('image_guidance_scale', 2.0),
                    cfg_range=self.config.rollout.get('cfg_range', (0.0, 1.0)),
                )
            
            print(f"✅ Method 2 completed")
            print(f"   Recomputed log probs shape: {log_probs_method2.shape}")
            print(f"   Recomputed log probs range: {log_probs_method2.min().item():.4f} to {log_probs_method2.max().item():.4f}")
            
            # Move Method 2 results to CPU for comparison
            log_probs_method2 = log_probs_method2.cpu()
            
            # Clear GPU cache again
            torch.cuda.empty_cache()
            print("   Moved results to CPU and cleared GPU cache")
            
            return True, "Both methods completed successfully", log_probs_method1, log_probs_method2
            
        except Exception as e:
            import traceback
            return False, f"Test failed: {str(e)}\n{traceback.format_exc()}", None, None

def analyze_results(log_probs_method1, log_probs_method2):
    """Analyze and compare the results from both methods"""
    print(f"\n📊 Comparing results...")
    
    # Check shapes match
    if log_probs_method1.shape != log_probs_method2.shape:
        print(f"❌ Shape mismatch!")
        print(f"   Method 1 shape: {log_probs_method1.shape}")
        print(f"   Method 2 shape: {log_probs_method2.shape}")
        return False
    
    # Compute differences
    abs_diff = torch.abs(log_probs_method1 - log_probs_method2)
    rel_diff = abs_diff / (torch.abs(log_probs_method1) + 1e-8)
    
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    print(f"📈 Difference Statistics:")
    print(f"   Max absolute difference: {max_abs_diff:.8f}")
    print(f"   Mean absolute difference: {mean_abs_diff:.8f}")
    print(f"   Max relative difference: {max_rel_diff:.8f}")
    print(f"   Mean relative difference: {mean_rel_diff:.8f}")
    
    # Print step-by-step comparison
    print(f"\n📋 Step-by-step comparison (first sample):")
    batch_idx = 0
    for step in range(min(10, log_probs_method1.shape[1])):
        val1 = log_probs_method1[:, step]
        val2 = log_probs_method2[:, step]
        diff = abs(val1 - val2)
        print(f"   Step {step:2d}: Method1={val1}, Method2={val2}, Diff={diff}")
    
    # Print trend analysis
    print(f"\n📈 Trend Analysis (Method 1):")
    method1_first = log_probs_method1[batch_idx, 0].item()
    method1_last = log_probs_method1[batch_idx, -1].item()
    method1_trend = "increasing" if method1_last > method1_first else "decreasing"
    print(f"   First step: {method1_first:.4f}")
    print(f"   Last step: {method1_last:.4f}")
    print(f"   Overall trend: {method1_trend}")
    
    print(f"\n📈 Trend Analysis (Method 2):")
    method2_first = log_probs_method2[batch_idx, 0].item()
    method2_last = log_probs_method2[batch_idx, -1].item()
    method2_trend = "increasing" if method2_last > method2_first else "decreasing"
    print(f"   First step: {method2_first:.4f}")
    print(f"   Last step: {method2_last:.4f}")
    print(f"   Overall trend: {method2_trend}")
    
    # Determine if test passes
    tolerance_abs = 1e-5
    tolerance_rel = 1e-4
    
    test_passed = (max_abs_diff < tolerance_abs and max_rel_diff < tolerance_rel)
    
    if test_passed:
        print(f"\n✅ TEST PASSED!")
        print(f"   Both methods produce nearly identical log probabilities")
        print(f"   (within tolerances: abs={tolerance_abs}, rel={tolerance_rel})")
    else:
        print(f"\n❌ TEST FAILED!")
        print(f"   Methods produce different log probabilities")
        print(f"   Exceeded tolerances: abs={tolerance_abs}, rel={tolerance_rel}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"log_prob_comparison_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("Log Probability Comparison Test Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("\n")
        
        f.write("Method 1 (Generation) Log Probs:\n")
        for step in range(log_probs_method1.shape[1]):
            val = log_probs_method1[batch_idx, step].item()
            f.write(f"  Step {step:2d}: {val:.8f}\n")
        
        f.write("\nMethod 2 (Recomputation) Log Probs:\n")
        for step in range(log_probs_method2.shape[1]):
            val = log_probs_method2[batch_idx, step].item()
            f.write(f"  Step {step:2d}: {val:.8f}\n")
        
        f.write("\nDifferences:\n")
        for step in range(log_probs_method1.shape[1]):
            val1 = log_probs_method1[batch_idx, step].item()
            val2 = log_probs_method2[batch_idx, step].item()
            diff = abs(val1 - val2)
            f.write(f"  Step {step:2d}: {diff:.8f}\n")
        
        f.write(f"\nStatistics:\n")
        f.write(f"  Max absolute difference: {max_abs_diff:.8f}\n")
        f.write(f"  Mean absolute difference: {mean_abs_diff:.8f}\n")
        f.write(f"  Max relative difference: {max_rel_diff:.8f}\n")
        f.write(f"  Mean relative difference: {mean_rel_diff:.8f}\n")
        f.write(f"  Test result: {'PASSED' if test_passed else 'FAILED'}\n")
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    return test_passed

def main():
    """Main test function"""
    print("🧪 Starting Log Probability Comparison Test...")
    print("=" * 60)
    
    # Set initial random seed
    set_random_seed(42)
    
    # Set up environment
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('LOCAL_RANK', '0')
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12347')
    
    # Create configuration
    config = create_test_config()
    print(f"📡 Using model: {config.model.path}")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=4,
            num_gpus=1 if torch.cuda.is_available() else 0
        )
    
    try:
        print("🚀 Initializing test worker...")
        worker = LogProbTestWorker.remote(config)
        
        # Initialize model
        print("\n🔧 Initializing OmniGen2 model...")
        success, message = ray.get(worker.init_model.remote())
        if not success:
            print(f"❌ {message}")
            return False
        print(f"✅ {message}")
        
        # Get tokenizer
        print("\n📝 Getting tokenizer...")
        success, tokenizer = ray.get(worker.get_tokenizer.remote())
        if not success:
            print(f"❌ {tokenizer}")
            return False
        print("✅ Tokenizer obtained")
        
        # Create test prompt
        prompt = "A cute cat sitting on a windowsill"
        prompts = [prompt, 'a dog']
        print(f"\n📋 Testing with prompt: '{prompt}'")
        
        # Create prompt data
        print("\n🔄 Processing prompt...")
        prompt_data = create_prompt_data_proto(prompts, tokenizer)
        print(f"✅ Created DataProto with batch size: {prompt_data.batch.batch_size}")
        
        # Run the comparison test
        print(f"\n🧪 Running log probability comparison test...")
        success, message, log_probs_method1, log_probs_method2 = ray.get(
            worker.test_log_prob_comparison.remote(prompt_data)
        )
        
        if not success:
            print(f"❌ {message}")
            return False
        
        print(f"✅ {message}")
        
        # Analyze results
        if log_probs_method1 is not None and log_probs_method2 is not None:
            test_passed = analyze_results(log_probs_method1, log_probs_method2)
            
            if test_passed:
                print(f"\n🎉 All tests passed!")
                print(f"   Log probability calculations are consistent between methods")
            else:
                print(f"\n⚠️  Tests revealed inconsistencies!")
                print(f"   Log probability calculations differ between methods")
                
            return test_passed
        else:
            print(f"❌ Could not obtain log probabilities for comparison")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up Ray
        if ray.is_initialized():
            ray.shutdown()
            print("🧹 Ray shutdown complete")
    
    print("\n" + "=" * 60)
    print("🏁 Log probability comparison test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 