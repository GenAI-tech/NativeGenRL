#!/usr/bin/env python3
"""
Test script for OmniGen2 Image Generation through ActorRolloutRefWorker.
This script tests both OmniGen2Pipeline and OmniGen2ChatPipeline with text prompts and saves generated images.
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

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from tensordict import TensorDict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_generation_config():
    """Create a configuration for OmniGen2 image generation testing"""
    config = OmegaConf.create({
        'use_chat_pipeline': False, # True: Use OmniGen2ChatPipeline for text first then image generation
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
            'ulysses_sequence_parallel_size': 1
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
            'height': 384,
            'width': 384,
            'num_inference_steps': 50,
            'scheduler': 'euler'  # Options: 'euler', 'dpmsolver++'
        },
        'ref': {
            'log_prob_micro_batch_size_per_gpu': 4,
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

def create_image_generation_prompts():
    """Create text prompts for image generation testing"""
    prompts = [
        # "A majestic mountain landscape at sunset with snow-capped peaks",
        # "A cute corgi dog sitting in a field of colorful flowers",
        # "A futuristic city skyline with flying cars and neon lights", 
        # "A serene beach scene with palm trees and crystal clear water",
        "A cozy coffee shop interior with warm lighting and books"
    ]
    return prompts

def create_prompt_data_proto(prompts, tokenizer):
    """Create DataProto from text prompts for image generation"""
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
    
    # Create position ids as torch tensor
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Create batch data with torch tensors
    batch_data = TensorDict({
        'input_ids': input_ids,
        'attention_mask': attention_mask, 
        'position_ids': position_ids,
    }, batch_size=batch_size)
    
    # Create meta info for generation
    meta_info = {
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'image_start_token_id': tokenizer.pad_token_id,  # Fallback for OmniGen2
        'do_sample': True,  # Use deterministic generation for testing
        'response_length': 128,
        'temperature': 1.0,
        'device': device  # Add device info to meta_info
    }
    
    # Add raw prompts and raw_prompt_ids as numpy arrays for non_tensor_batch
    raw_prompt_ids = [
        tokenizer.encode(prompt, add_special_tokens=False) 
        for prompt in prompts
    ]
    non_tensor_data = {
        # 'raw_prompt': prompts,  # assert isinstance(val, np.ndarray)
        # 'images': np.array([], dtype=object),
        'raw_prompt_ids': np.array(raw_prompt_ids, dtype=object)
    }
    
    return DataProto(batch=batch_data, meta_info=meta_info, non_tensor_batch=non_tensor_data)

def save_generated_results(result, prompts, use_chat_pipeline=False, output_dir="generated_results"):
    """Save generated images, texts (for chat pipeline), and log probabilities
    
    Args:
        result: DataProto containing generation results
        prompts: Original prompts used for generation
        use_chat_pipeline: Whether chat pipeline was used
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    
    # Extract images
    images = result.batch['gen_img'].cpu().numpy()
    
    # Extract log probabilities if available
    log_probs = None
    if 'log_probs' in result.batch:
        log_probs = result.batch['log_probs']
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.cpu().numpy()
        print(f"   Found log probabilities with shape: {log_probs.shape if hasattr(log_probs, 'shape') else len(log_probs)}")
    
    # Extract generated texts if available (from chat pipeline)
    generated_texts = None
    if hasattr(result, 'non_tensor_batch') and 'generated_texts' in result.non_tensor_batch:
        generated_texts = result.non_tensor_batch['generated_texts']
    
    # Handle case where there are multiple images per prompt
    num_images = len(images)
    num_prompts = len(prompts)
    images_per_prompt = num_images // num_prompts if num_prompts > 0 else 1
    
    print(f"💾 Saving {num_images} generated images{'and texts' if use_chat_pipeline else ''}...")
    
    for i, img_array in enumerate(images):
        # Determine which prompt this image corresponds to
        prompt_idx = i // images_per_prompt if images_per_prompt > 0 else i % num_prompts
        prompt = prompts[prompt_idx] if prompt_idx < len(prompts) else f"prompt_{prompt_idx}"
        
        # Create safe filename from prompt
        safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in prompt)
        safe_prompt = safe_prompt.replace(' ', '_')[:50]  # Limit length
        
        # Save image
        img = PIL.Image.fromarray(img_array.astype('uint8'))
        if images_per_prompt > 1:
            img_filename = f"{safe_prompt}_{i%images_per_prompt+1}_{timestamp}.png"
        else:
            img_filename = f"{safe_prompt}_{timestamp}.png"
        
        img_path = os.path.join(output_dir, img_filename)
        img.save(img_path)
        saved_files.append(img_path)
        
        # Save detailed information file (text response + log probs)
        if images_per_prompt > 1:
            info_filename = f"{safe_prompt}_{i%images_per_prompt+1}_{timestamp}.txt"
        else:
            info_filename = f"{safe_prompt}_{timestamp}.txt"
        
        info_path = os.path.join(output_dir, info_filename)
        
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"Pipeline Type: {'Chat Pipeline' if use_chat_pipeline else 'Standard Pipeline'}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("="*60 + "\n")
            f.write(f"Original Prompt: {prompt}\n")
            f.write("="*60 + "\n")
            
            # Add text response if available (chat pipeline)
            if use_chat_pipeline:
                if generated_texts is not None and i < len(generated_texts):
                    f.write(f"Generated Text: {generated_texts[i]}\n")
                else:
                    f.write("Generated Text: [Image generated without text response]\n")
                f.write("="*60 + "\n")
            
            # Add log probabilities information if available
            if log_probs is not None:
                f.write("Log Probabilities Information:\n")
                if log_probs is not None and len(log_probs) > i:
                    current_log_probs = log_probs[i]
                    if isinstance(current_log_probs, (list, np.ndarray)):
                        f.write(f"Log probabilities: {[float(x) for x in current_log_probs]}\n")
                    else:
                        f.write(f"Log probability: {float(current_log_probs)}\n")
                else:
                    f.write("Log Probabilities: Not available\n")
                f.write("="*60 + "\n")
        
        saved_files.append(info_path)
        
        print(f"  ✅ Saved: {img_filename} and {info_filename}")
    
    return saved_files

@ray.remote(num_gpus=1)
class ImageGenerationWorker:
    """Ray remote worker for testing image generation"""
    
    def __init__(self, config):
        self.config = config
        self.worker = None
        
    def init_model(self):
        """Initialize the model and check which pipeline type was loaded"""
        try:
            # Ensure we're using CUDA if available
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                device = torch.cuda.current_device()
                print(f"Using CUDA device: {device}")
            
            self.worker = ActorRolloutRefWorker(self.config, role='actor_rollout')  # Need both actor and rollout
            self.worker.init_model()
            
            # Check which pipeline type was loaded
            pipeline_type = "Unknown"
            is_chat_pipeline = False
            
            try:
                from omnigen2.pipelines.omnigen2.pipeline_omnigen2_chat import OmniGen2ChatPipeline
                from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
                
                if isinstance(self.worker.actor_module_fsdp, OmniGen2ChatPipeline):
                    pipeline_type = "OmniGen2ChatPipeline"
                    is_chat_pipeline = True
                elif isinstance(self.worker.actor_module_fsdp, OmniGen2Pipeline):
                    pipeline_type = "OmniGen2Pipeline"
                    is_chat_pipeline = False
                else:
                    pipeline_type = f"Unknown pipeline type: {type(self.worker.actor_module_fsdp)}"
                    
            except Exception as e:
                pipeline_type = f"Could not determine pipeline type: {e}"
            
            # Ensure the model components are on the correct device
            if hasattr(self.worker, 'actor_module_fsdp') and torch.cuda.is_available():
                # The model should already be on GPU, but let's make sure
                print(f"Model initialized successfully - Pipeline: {pipeline_type}")
                
            return True, f"Model initialized successfully - Pipeline: {pipeline_type}", is_chat_pipeline
        except Exception as e:
            import traceback
            return False, f"Model initialization failed: {str(e)}\n{traceback.format_exc()}", False
    
    def generate_images(self, prompt_data_proto):
        """Generate images from text prompts"""
        try:
            if self.worker is None:
                return False, "Worker not initialized"
                
            # Ensure input data is on the correct device
            device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
            print(f"Moving input data to device: {device}")
            
            # Move the DataProto to the correct device
            prompt_data_proto = prompt_data_proto.to(device)
            
            # Use the generate_sequences method
            result = self.worker.generate_sequences(prompt_data_proto)
            return True, result
        except Exception as e:
            import traceback
            return False, f"Image generation failed: {str(e)}\n{traceback.format_exc()}"
    
    def get_tokenizer(self):
        """Get the tokenizer for prompt processing"""
        try:
            if self.worker and hasattr(self.worker, 'tokenizer'):
                return True, self.worker.tokenizer
            else:
                return False, "Tokenizer not available"
        except Exception as e:
            return False, f"Failed to get tokenizer: {str(e)}"

def main():
    """Main test function"""
    print("🎨 Starting OmniGen2 Image Generation Test...")
    
    # Set up distributed environment variables for testing
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('LOCAL_RANK', '0')
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12346')  # Different port from other tests
    
    # Create configuration
    config = create_generation_config()
    print(f"📡 Using model: {config.model.path}")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=4,
            num_gpus=1 if torch.cuda.is_available() else 0
        )
    
    # try:
    print("🚀 Initializing image generation worker...")
    worker = ImageGenerationWorker.remote(config)
    
    # Initialize model
    print("\n🔧 Initializing OmniGen2 model...")
    success, message, is_chat_pipeline = ray.get(worker.init_model.remote())
    if not success:
        print(f"❌ {message}")
        return
    print(f"✅ {message}")
    
    # Get tokenizer
    print("\n📝 Getting tokenizer...")
    success, tokenizer = ray.get(worker.get_tokenizer.remote())
    if not success:
        print(f"❌ {tokenizer}")
        return
    print("✅ Tokenizer obtained")
    
    # Create test prompts
    prompts = create_image_generation_prompts()
    print(f"\n📋 Created {len(prompts)} test prompts:")
    for i, prompt in enumerate(prompts, 1):
        print(f"   {i}. {prompt}")
    
    # Create prompt data
    print("\n🔄 Processing prompts...")
    prompt_data = create_prompt_data_proto(prompts, tokenizer)
    print(f"prompt_data: {prompt_data}")
    print(f"✅ Created DataProto with batch size: {prompt_data.batch.batch_size}")
    
    # Generate images
    print("\n🎨 Generating images...")
    print("   This may take a few minutes depending on GPU and model size...")
    success, result = ray.get(worker.generate_images.remote(prompt_data))
    
    if not success:
        print(f"❌ {result}")
        return
        
    print("✅ Generation completed successfully!")
    
    # Extract and save generated images
    print("\n💾 Saving generated images...")
    if 'gen_img' in result.batch:
        generated_images = result.batch['gen_img']
        print(f"   Found {len(generated_images)} generated images")
        print(f"   Image shape: {generated_images[0].shape if len(generated_images) > 0 else 'N/A'}")
        
        # Check for log probabilities
        log_probs_available = False
        if 'log_probs' in result.batch:
            log_probs = result.batch['log_probs']
            log_probs_available = True
            print(f"   Found log probabilities: {type(log_probs)}")
            
            # Display log probabilities statistics
            if isinstance(log_probs, torch.Tensor):
                log_probs_np = log_probs.cpu().numpy()
                print(f"   Log probs tensor shape: {log_probs_np.shape}")
            elif isinstance(log_probs, (list, np.ndarray)):
                log_probs_np = np.array(log_probs) if not isinstance(log_probs, np.ndarray) else log_probs
                print(f"   Log probs array length: {len(log_probs_np)}")

        
        # Check for generated texts (chat pipeline)
        generated_texts = None
        if hasattr(result, 'non_tensor_batch') and 'generated_texts' in result.non_tensor_batch:
            generated_texts = result.non_tensor_batch['generated_texts']
            print(f"   Found {len(generated_texts)} generated text responses")
        
        # Save results
        saved_files = save_generated_results(result, prompts, config.use_chat_pipeline)
        
        # Print summary
        print(f"\n🎉 Successfully saved {len(saved_files)} files!")
        print("📁 Saved files:")
        for f in saved_files:
            print(f"   - {f}")
        
        # Print pipeline-specific summary
        print(f"   • Images generated: {len(generated_images)}")
        if log_probs_available:
            print(f"   • Log probabilities: Available (saved in .txt files)")
        if generated_texts is not None:
            print(f"   • Text responses: {len(generated_texts)}")
            print("   • Sample text responses:")
            for i, text in enumerate(generated_texts[:3]):
                print(f"     {i+1}: {text[:80]}{'...' if len(text) > 80 else ''}")
            
    else:
        print("⚠️  No 'gen_img' found in result batch")
        print(f"Available keys: {list(result.batch.keys())}")
    
    print("\n✨ Image generation test completed successfully!")
        
    # except Exception as e:
    #     print(f"❌ Test failed with error: {str(e)}")
    #     import traceback
    #     traceback.print_exc()
    
    # finally:
    #     # Clean up Ray
    #     if ray.is_initialized():
    #         ray.shutdown()
    #         print("🧹 Ray shutdown complete")

if __name__ == "__main__":
    print("=" * 60)
    print("🖼️  OmniGen2 Image Generation Test")
    print("=" * 60)
    
    main()
    
    print("\n" + "=" * 60)
    print("🏁 Image generation test completed!")
    print("=" * 60) 