# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""
import contextlib
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask
from .base import BaseRollout

from transformers import GenerationConfig
import numpy as np
from typing import Union, List, Any
import random
import PIL.Image

__all__ = ['HFRollout']
NEGATIVE_PROMPT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

class HFRollout(BaseRollout):

    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module
        self.cot_generate = config.get('cot_generate', False)

        # Check if this is an OmniGen2Pipeline
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2_chat import OmniGen2ChatPipeline
        self.is_omnigen2 = isinstance(module, OmniGen2Pipeline) or isinstance(module, OmniGen2ChatPipeline)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get('micro_batch_size', batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output
    

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        if self.is_omnigen2:
            return self._generate_omnigen2(prompts)
        else:
            return self._generate_standard(prompts)

    def _generate_omnigen2(self, prompts: DataProto) -> DataProto:
        """Generate using OmniGen2Pipeline"""
        # Extract prompts from non_tensor_batch if available, otherwise construct from input_ids
        if hasattr(prompts, 'non_tensor_batch') and 'raw_prompt' in prompts.non_tensor_batch:
            text_prompts = prompts.non_tensor_batch['raw_prompt']
        else:
            # Decode input_ids to get text prompts
            text_prompts = self.module.processor.tokenizer.batch_decode(
                prompts.batch['input_ids'], skip_special_tokens=True
            )
        
        batch_size = len(text_prompts)

        if hasattr(prompts, 'non_tensor_batch') and 'images' in prompts.non_tensor_batch:
            input_images = prompts.non_tensor_batch['images']
        else:
            if batch_size > 1:
                input_images = [None] * batch_size
            else:
                input_images = None
        
        # Get generation parameters
        do_sample = prompts.meta_info.get('do_sample', True)
        response_length = prompts.meta_info.get('response_length', self.config.response_length)
        temperature = prompts.meta_info.get('temperature', self.config.temperature)
        is_validate = prompts.meta_info.get('validate', False)
        
        # Set generation parameters with proper defaults
        text_guidance_scale = self.config.get('text_guidance_scale', 5.0)  # Updated default to match official
        image_guidance_scale = self.config.get('image_guidance_scale', 2.0)  # Updated default to match official
        height = self.config.get('height', 1024)
        width = self.config.get('width', 1024)
        num_inference_steps = self.config.get('num_inference_steps', 28)
        cfg_range = self.config.get('cfg_range', (0.0, 1.0))  # New parameter
        max_input_image_side_length = self.config.get('max_input_image_side_length', 2048)  # New parameter
        max_pixels = self.config.get('max_pixels', 1024*1024)  # New parameter
        noise_level = self.config.get('noise_level', 0.3)  # New parameter
        max_sequence_length = self.config.get('max_sequence_length', 1024)  # Fixed sequence length
        num_images_per_prompt = self.config.get('n', 1)
        if not do_sample:
            text_guidance_scale = 1.0
            image_guidance_scale = 1.0
            num_images_per_prompt = 1

        elif is_validate:
            text_guidance_scale = self.config.val_kwargs.get('text_guidance_scale', text_guidance_scale)
            image_guidance_scale = self.config.val_kwargs.get('image_guidance_scale', image_guidance_scale)
            height = self.config.val_kwargs.get('height', height)
            width = self.config.val_kwargs.get('width', width)
            num_inference_steps = self.config.val_kwargs.get('num_inference_steps', num_inference_steps)
            cfg_range = self.config.val_kwargs.get('cfg_range', cfg_range)
            max_input_image_side_length = self.config.val_kwargs.get('max_input_image_side_length', max_input_image_side_length)
            max_pixels = self.config.val_kwargs.get('max_pixels', max_pixels)
            noise_level = self.config.val_kwargs.get('noise_level', noise_level)
            max_sequence_length = self.config.val_kwargs.get('max_sequence_length', max_sequence_length)
            num_images_per_prompt = 1
        
        # Handle multiple responses per prompt
        original_batch_size = batch_size
        if num_images_per_prompt > 1:
            # Use num_images_per_prompt instead of repeating prompts
            # Total output batch size will be original_batch_size * num_images_per_prompt
            expected_output_batch_size = original_batch_size * num_images_per_prompt
        else:
            expected_output_batch_size = batch_size
        print(f"expected_output_batch_size: {expected_output_batch_size}, batch_size: {batch_size}, num_images_per_prompt: {num_images_per_prompt}")

        if self.config.get('seed', -1) == -1:
            seed_input = random.randint(0, 2**16 - 1)
        else:
            seed_input = self.config.seed
        generator = torch.Generator(device=self.module.device).manual_seed(seed_input)


        # Generate using OmniGen2Pipeline or OmniGen2ChatPipeline
        try:
            print(f"Generating {batch_size} images with prompts: {text_prompts[:2]}...")  # Show first 2 prompts
            
            # Check if this is a chat pipeline
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2_chat import OmniGen2ChatPipeline
            is_chat_pipeline = isinstance(self.module, OmniGen2ChatPipeline)
            
            if is_chat_pipeline:
                # Handle OmniGen2ChatPipeline - returns (text, images) tuple
                print("Using OmniGen2ChatPipeline for multimodal generation...")
                
                # For chat pipeline, we process one prompt at a time since it's designed for single conversations
                all_generated_images = []
                all_generated_texts = []
                
                for prompt in text_prompts:
                    result = self.module(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        input_images=input_images[0] if input_images and len(input_images) > 0 else None,
                        height=height,
                        width=width,
                        num_images_per_prompt=num_images_per_prompt,
                        num_inference_steps=num_inference_steps,
                        text_guidance_scale=text_guidance_scale,
                        image_guidance_scale=image_guidance_scale,
                        cfg_range=cfg_range,
                        max_sequence_length=max_sequence_length,
                        max_pixels=max_pixels,
                        generator=generator,
                        noise_level=noise_level,
                        return_dict=True
                    )
                    
                    # Extract text and images from chat pipeline result
                    generated_text = result.text if hasattr(result, 'text') else ""
                    generated_images_batch = result.images if hasattr(result, 'images') and result.images is not None else []
                    # print(f"prompt: {prompt}, text: {generated_text}, generated_images_batch: {generated_images_batch.shape}")
                    all_generated_texts.append(generated_text)
                    
                    # Handle the case where no images are generated (text-only response)
                    if not generated_images_batch:
                        # Create dummy images for consistency - one for each expected image per prompt
                        for _ in range(num_images_per_prompt):
                            dummy_img = PIL.Image.new('RGB', (height, width), color='black')
                            all_generated_images.append(dummy_img)
                    else:
                        # Add generated images
                        all_generated_images.extend(generated_images_batch)
                
                generated_images = all_generated_images
                generated_texts = all_generated_texts
                diffusion_data = None  # Chat pipeline doesn't provide diffusion data
                
            else:
                # Handle standard OmniGen2Pipeline - returns images only
                print("Using standard OmniGen2Pipeline for image generation...")
                
                result = self.module(
                    prompt=text_prompts,
                    negative_prompt=NEGATIVE_PROMPT,
                    input_images=input_images,
                    height=height,
                    width=width,
                    num_images_per_prompt=num_images_per_prompt,
                    num_inference_steps=num_inference_steps,
                    text_guidance_scale=text_guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    cfg_range=cfg_range,
                    max_sequence_length=max_sequence_length,
                    max_pixels=max_pixels,
                    max_input_image_side_length=max_input_image_side_length,
                    generator=generator,
                    noise_level=noise_level,
                    return_dict=True
                )
                
                # Handle different return types for standard pipeline
                if result is None:
                    print("Warning: OmniGen2Pipeline returned None, creating dummy images")
                    generated_images = [PIL.Image.new('RGB', (height, width), color='black') for _ in range(expected_output_batch_size)]
                    log_probs = None
                    diffusion_data = None
                elif hasattr(result, 'images'):
                    generated_images = result.images
                    log_probs = result.log_probs
                    diffusion_data = result.diffusion_data
                else:
                    print(f"Warning: Unexpected result type {type(result)}, creating dummy images")
                    generated_images = [PIL.Image.new('RGB', (height, width), color='black') for _ in range(expected_output_batch_size)]
                    log_probs = None
                    diffusion_data = None
                
                generated_texts = None  # Standard pipeline doesn't generate text
                
        except Exception as e:
            print(f"Error in OmniGen2 pipeline generation: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: create dummy images
            generated_images = [PIL.Image.new('RGB', (height, width), color='black') for _ in range(expected_output_batch_size)]
            generated_texts = None

        # Safety check: ensure we have the correct number of images
        if len(generated_images) != expected_output_batch_size:
            print(f"Warning: Generated {len(generated_images)} images but expected {expected_output_batch_size}. Adjusting...")
            if len(generated_images) < expected_output_batch_size:
                # Add dummy images to reach expected size
                while len(generated_images) < expected_output_batch_size:
                    dummy_img = PIL.Image.new('RGB', (height, width), color='black')
                    generated_images.append(dummy_img)
            else:
                # Truncate to expected size
                generated_images = generated_images[:expected_output_batch_size]
            print(f"Adjusted to {len(generated_images)} images")

        # Convert PIL images to numpy arrays
        gen_img_arrays = []
        for i, img in enumerate(generated_images):
            if img is None:
                print(f"Warning: Image {i} is None, creating dummy image")
                img = PIL.Image.new('RGB', (1024, 1024), color='black')
            img_array = np.array(img)  # Shape: (H, W, 3)
            gen_img_arrays.append(img_array)
        
        gen_img_tensor = torch.stack([torch.from_numpy(arr) for arr in gen_img_arrays])  # (B, H, W, 3)
        
        # Create response tokens - for image generation, we can use image placeholder tokens
        eos_token_id = prompts.meta_info['eos_token_id']
        pad_token_id = prompts.meta_info['pad_token_id']
        image_start_token_id = prompts.meta_info.get('image_start_token_id', pad_token_id)
        
        # Create dummy response tokens for compatibility (image generation doesn't produce text)
        device = gen_img_tensor.device
        response_tokens = torch.full((expected_output_batch_size, response_length), pad_token_id, dtype=torch.long, device=device)
        # Set first token to image_start_token to indicate image generation
        response_tokens[:, 0] = image_start_token_id
        # Set last token to EOS
        response_tokens[:, -1] = eos_token_id

        # Get original prompt info or create dummy
        if 'input_ids' in prompts.batch:
            original_input_ids = prompts.batch['input_ids'].to(device)
            original_attention_mask = prompts.batch['attention_mask'].to(device)
            original_position_ids = prompts.batch['position_ids'].to(device)
            
            # Repeat prompt info for multiple images per prompt
            if num_images_per_prompt > 1:
                original_input_ids = original_input_ids.repeat_interleave(num_images_per_prompt, dim=0)
                original_attention_mask = original_attention_mask.repeat_interleave(num_images_per_prompt, dim=0)
                original_position_ids = original_position_ids.repeat_interleave(num_images_per_prompt, dim=0)
        else:
            # Create dummy inputs for compatibility
            dummy_prompt_length = 10
            original_input_ids = torch.full((expected_output_batch_size, dummy_prompt_length), pad_token_id, dtype=torch.long, device=device)
            original_attention_mask = torch.ones_like(original_input_ids)
            original_position_ids = torch.arange(dummy_prompt_length, device=device).unsqueeze(0).expand(expected_output_batch_size, -1)
        
        # Concatenate original prompts with responses
        sequences = torch.cat([original_input_ids, response_tokens], dim=-1)
        
        # Create attention mask for full sequence
        full_attention_mask = torch.cat([
            original_attention_mask,
            torch.ones((expected_output_batch_size, response_length), dtype=torch.long, device=device)
        ], dim=-1)
        
        # Create position ids for full sequence
        max_pos = original_position_ids.max(dim=-1)[0] + 1
        response_position_ids = torch.arange(response_length, device=device).unsqueeze(0).expand(expected_output_batch_size, -1)
        response_position_ids = response_position_ids + max_pos.unsqueeze(-1)
        full_position_ids = torch.cat([original_position_ids, response_position_ids], dim=-1)
        
        # Final validation: ensure all tensors have correct batch dimensions
        assert sequences.shape[0] == expected_output_batch_size, f"sequences batch size mismatch: {sequences.shape[0]} vs {expected_output_batch_size}"
        assert full_attention_mask.shape[0] == expected_output_batch_size, f"attention_mask batch size mismatch: {full_attention_mask.shape[0]} vs {expected_output_batch_size}"
        assert full_position_ids.shape[0] == expected_output_batch_size, f"position_ids batch size mismatch: {full_position_ids.shape[0]} vs {expected_output_batch_size}"
        assert response_tokens.shape[0] == expected_output_batch_size, f"response_tokens batch size mismatch: {response_tokens.shape[0]} vs {expected_output_batch_size}"
        assert gen_img_tensor.shape[0] == expected_output_batch_size, f"gen_img_tensor batch size mismatch: {gen_img_tensor.shape[0]} vs {expected_output_batch_size}"
        
        # Create output DataProto
        output_batch = TensorDict({
            'prompts': original_input_ids,
            'input_ids': sequences,
            'attention_mask': full_attention_mask,
            'position_ids': full_position_ids,
            'responses': response_tokens,
            'gen_img': gen_img_tensor,
        }, batch_size=expected_output_batch_size)

        # Add diffusion data if available (from standard pipeline)
        if diffusion_data is not None:
            # Add debug logging to identify dimension mismatches
            import os
            worker_id = os.getenv('RANK', 'unknown')
            print(f"Debug Worker {worker_id}: Adding diffusion data for batch_size={expected_output_batch_size}")
            
            # Print shapes of all diffusion tensors
            for key, value in diffusion_data.items():
                if torch.is_tensor(value):
                    print(f"Debug Worker {worker_id}: {key} shape: {value.shape}")
                elif isinstance(value, list):
                    print(f"Debug Worker {worker_id}: {key} is list with length: {len(value)}")
                    if len(value) > 0 and value[0] is not None:
                        if torch.is_tensor(value[0]):
                            print(f"Debug Worker {worker_id}: {key}[0] shape: {value[0].shape}")
                        elif isinstance(value[0], list) and len(value[0]) > 0:
                            print(f"Debug Worker {worker_id}: {key}[0] is nested list, first element shape: {value[0][0].shape if torch.is_tensor(value[0][0]) else type(value[0][0])}")
                else:
                    print(f"Debug Worker {worker_id}: {key} type: {type(value)}")
        
            # Add padded embeddings to batch
            if diffusion_data is not None:
                print(f"Debug Worker {worker_id}: Adding padded embeddings to batch")
                
                output_batch.update({
                   'latents': diffusion_data['latents'],
                    'next_latents': diffusion_data['next_latents'],
                    'timesteps': diffusion_data['timesteps'],
                    'ref_latents': diffusion_data['ref_latents'],
                    # 'prompt_embeds': prompt_embeds_padded,
                    # 'prompt_attention_mask': prompt_attention_mask_padded, 
                    'negative_prompt_embeds': diffusion_data['negative_prompt_embeds'],
                    'negative_prompt_attention_mask': diffusion_data['negative_prompt_attention_mask'],
                })
                print(f"Debug Worker {worker_id}: Successfully added embeddings to output_batch")
                for key, value in output_batch.items():
                    if torch.is_tensor(value):
                        print(f"  {key}: {value.shape}")
        # Add log probabilities if available
        if log_probs is not None:
            output_batch['log_probs'] = log_probs
        
        # Prepare non_tensor_batch data
        non_tensor_data = {}

        print('output_batch', output_batch)

        # Store variable-length embeddings in non_tensor_batch as object arrays
        # This allows each sample to have different sequence lengths
        prompt_embeds_list = [diffusion_data['prompt_embeds'][i].cpu().to(dtype=torch.float32) for i in range(diffusion_data['prompt_embeds'].shape[0])]
        prompt_attention_mask_list = [diffusion_data['prompt_attention_mask'][i].cpu().to(dtype=torch.float32) for i in range(diffusion_data['prompt_attention_mask'].shape[0])]
        non_tensor_data.update({
            'prompt_embeds': np.array(prompt_embeds_list, dtype=object),
            'prompt_attention_mask': np.array(prompt_attention_mask_list, dtype=object),
        })
        
        # Add generated texts if available (from chat pipeline)
        if generated_texts is not None:
            # Repeat texts for multiple images per prompt if needed
            if num_images_per_prompt > 1:
                expanded_texts = []
                for text in generated_texts:
                    expanded_texts.extend([text] * num_images_per_prompt)
                non_tensor_data['generated_texts'] = np.array(expanded_texts, dtype=object)
            else:
                non_tensor_data['generated_texts'] = np.array(generated_texts, dtype=object)
        
        # Add original prompts for reference
        if hasattr(prompts, 'non_tensor_batch') and 'raw_prompt' in prompts.non_tensor_batch:
            original_prompts = prompts.non_tensor_batch['raw_prompt']
            # Repeat prompts for multiple images per prompt if needed
            if num_images_per_prompt > 1:
                expanded_prompts = []
                for prompt in original_prompts:
                    expanded_prompts.extend([prompt] * num_images_per_prompt)
                non_tensor_data['raw_prompt'] = np.array(expanded_prompts, dtype=object)
            else:
                non_tensor_data['raw_prompt'] = np.array(original_prompts, dtype=object)
        
        # empty cache before compute old_log_prob
        torch.cuda.empty_cache()

        if non_tensor_data:
            return DataProto(batch=output_batch, meta_info=prompts.meta_info, non_tensor_batch=non_tensor_data)
        else:
            return DataProto(batch=output_batch, meta_info=prompts.meta_info)

    def _generate_standard(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']  # left-padded attention_mask
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']
        pad_token_id = prompts.meta_info['pad_token_id']
        image_start_token_id = prompts.meta_info['image_start_token_id']

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        # make sampling args can be overriden by inputs
        do_sample = prompts.meta_info.get('do_sample', True)
        response_length = prompts.meta_info.get('response_length', self.config.response_length)
        top_p = prompts.meta_info.get('top_p', self.config.get('top_p', 1.0))
        top_k = prompts.meta_info.get('top_k', self.config.get('top_k', 0))

        if top_k is None:
            top_k = 0
        top_k = max(0, top_k)  # to be compatible with vllm
        
        is_validate = prompts.meta_info.get('validate', False)
        
        temperature = prompts.meta_info.get('temperature', self.config.temperature)
        
        kwargs = {'top_p': top_p, 'top_k': top_k, 'temperature': temperature}
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': 0,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }
        kwargs.update(cfg_weight=self.config['cfg_weight'])
        

        generation_config = GenerationConfig(do_sample=do_sample)
        generation_config = generation_config.update(**kwargs)    
        
        if self.config.n > 1 and do_sample and not is_validate:
            idx = _repeat_interleave(idx, self.config.n)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            batch_size = idx.size(0)
            prompt_length = idx.size(1)    
        
        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if self.cot_generate:
                    output = self.module.text_img_generate(
                        input_ids=idx,
                        attention_mask=attention_mask,
                        do_sample=do_sample,
                        max_new_tokens=response_length,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        image_start_token_id=image_start_token_id,
                        generation_config=generation_config,
                        output_scores=False,  # this is potentially very large
                        return_dict_in_generate=True,
                        use_cache=True)
                    text_tokens = output.text_tokens
                else:
                    output = self.module.generate(
                        input_ids=idx,
                        attention_mask=attention_mask,
                        do_sample=do_sample,
                        max_new_tokens=response_length,
                        # max_length=max_length,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        generation_config=generation_config,
                        # renormalize_logits=True,
                        output_scores=False,  # this is potentially very large
                        return_dict_in_generate=True,
                        use_cache=True)
        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        seq_img_mask = output.seq_img_mask  # seq_img_mask tells you which positions inside response (and the preceding prompt) are image tokens.

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
            delta_seq_img_mask = torch.zeros(size=(batch_size, delta_length), device=seq.device, dtype=seq_img_mask.dtype)
            seq_img_mask = torch.cat((seq_img_mask, delta_seq_img_mask), dim=1)
            if self.cot_generate:
                delta_text_tokens = torch.ones(size=(batch_size, delta_length), device=text_tokens.device, dtype=text_tokens.dtype)
                delta_text_tokens = pad_token_id * delta_text_tokens
                text_tokens = torch.cat((text_tokens, delta_text_tokens), dim=1)

        assert seq.shape[1] == sequence_length

        prompt = seq[:, :prompt_length]  # (bs, prompt_length)
        response = seq[:, prompt_length:]  # (bs, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        if delta_length > 0:
            response_attention_mask[..., -delta_length:] = 0
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                'prompts': prompt,  # (B, P)	prompt
                'responses': response,  # (B, R)	suffix of seq containing both text and image tokens
                'input_ids': seq,  # (B, P + R)	prompt + generated text + image tokens (if any)
                'attention_mask': attention_mask,  # (B, P + R) 1 → “real token”, 0 → padding; constructed so every real token—including image tokens—is attendable
                'position_ids': position_ids,
                'gen_img': output.gen_img,
                'seq_img_mask': seq_img_mask # (B, P + R)	1 → token is an image placeholder, 0 → normal text token
            },
            batch_size=batch_size)
        if self.cot_generate:
            batch['text_tokens'] = text_tokens

        # empty cache before compute old_log_prob
        torch.cuda.empty_cache()

        self.module.train()
        return DataProto(batch=batch)