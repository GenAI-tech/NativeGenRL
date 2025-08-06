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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
import numpy as np
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
from verl.utils.adaptive_entropy_coeff import AdaptiveEntropyCoefficient
__all__ = ['DataParallelPPOActor', 'DataParallelDiffusionPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.cfg_weight = self.config.get('cfg_weight', 5.0)
        self.detach_uncond = self.config.get('detach_uncond', False)
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        if self.config.get('adaptive_entropy_coeff',{}).get('enable', False): # only used for actor
            self.use_adaptive_entropy_coeff = True
            self.text_adaptive_entropy_coeff = AdaptiveEntropyCoefficient(
                initial_alpha = self.config.adaptive_entropy_coeff.text.get('initial_alpha', 0.0),
                target_entropy = self.config.adaptive_entropy_coeff.text.get('target_entropy', -1.0),
                lr = self.config.adaptive_entropy_coeff.text.get('lr', 1e-3),
                max_coeff= self.config.adaptive_entropy_coeff.text.get('max_coeff', 1e-3),
                min_coeff= self.config.adaptive_entropy_coeff.text.get('min_coeff', -1e-3),
            )
            self.img_adaptive_entropy_coeff = AdaptiveEntropyCoefficient(
                initial_alpha = self.config.adaptive_entropy_coeff.image.get('initial_alpha', 0.0),
                target_entropy = self.config.adaptive_entropy_coeff.image.get('target_entropy', -1.0),
                lr = self.config.adaptive_entropy_coeff.image.get('lr', 1e-3),
                max_coeff= self.config.adaptive_entropy_coeff.image.get('max_coeff', 1e-3),
                min_coeff= self.config.adaptive_entropy_coeff.image.get('min_coeff', -1e-3),
            )
        else:
            self.use_adaptive_entropy_coeff = False
               
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        print(f'Actor cfg_weight={self.cfg_weight}')
        print(f'Actor detach_uncond={self.detach_uncond}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get('use_torch_compile', True)  #  use torch compile by default
            else verl_F.entropy_from_logits)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            prompts = micro_batch['prompts']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                # this is for transformer AR next token prediction
                output = self.actor_module(input_ids=input_ids,
                                        #    input_img_mask=micro_batch['seq_img_mask'],
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           bos_token_id=self.config.bos_token_id,
                                           pad_token_id=self.config.pad_token_id,
                                           image_start_token_id=self.config.image_start_token_id,
                                           cfg_weight = self.cfg_weight,
                                           detach_uncond = self.detach_uncond,
                                           use_cache=False)  # prevent model thinks we are generating
        
                logits = output['logits'] if isinstance(output, dict) else output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                # clamp logits to avoid overflow
                logits = torch.clamp(logits, min=-30.0, max=30.0)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        if hasattr(self.actor_module, 'eval'):
            self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'seq_img_mask']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'seq_img_mask']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs', 'uid']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['uid']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
            # dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        # Handle update_mode filtering
        update_mode = getattr(self.config, 'update_mode', 'all')
        print(f"update_mode: {update_mode}")
        
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                # if has_multi_modal_inputs:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                # elif self.config.use_dynamic_bsz:
                #     max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                #     micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                # else:
                #     self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                #     # split batch into micro_batches
                #     micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    seq_img_mask = data['seq_img_mask']
                    response_mask = attention_mask[:, -response_length:]
                    
                    if update_mode == 'text':
                        # Only keep text tokens (non-image tokens)
                        text_mask = ~seq_img_mask[..., -response_length:]
                        response_mask = response_mask & text_mask
                    elif update_mode == 'image':
                        # Only keep image tokens
                        img_mask = seq_img_mask[..., -response_length:]
                        response_mask = response_mask & img_mask
                    elif update_mode == 'all':
                        pass
                    else:
                        raise ValueError(f"Invalid update_mode: {update_mode}. Must be 'text', 'image', or 'all'")
                    
                    # Keep current behavior - optionally ignore image start tokens
                    if self.config.ignore_img_start:
                        img_start_mask = responses == self.config.image_start_token_id
                        response_mask = response_mask & (~img_start_mask)
                    
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']
                    uids = data['uid']

                    clip_ratio = self.config.clip_ratio

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                  log_prob=log_prob,
                                                                                  advantages=advantages,
                                                                                  eos_mask=response_mask,
                                                                                  cliprange=clip_ratio,
                                                                                  uids=uids,
                                                                                  algo_name=self.config.algo_name)
                    # compute entropy loss from entropy
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)
                    text_entropy_loss = torch.tensor(0.0).to(entropy.device)
                    img_entropy_loss = torch.tensor(0.0).to(entropy.device)
                    if update_mode == 'text':
                        text_entropy_loss = verl_F.masked_mean(entropy, ~seq_img_mask[..., -response_length:] & response_mask)
                    elif update_mode == 'image':
                        img_entropy_loss = verl_F.masked_mean(entropy, seq_img_mask[..., -response_length:] & response_mask)
                    elif update_mode == 'all':
                        text_entropy_loss = verl_F.masked_mean(entropy, ~seq_img_mask[..., -response_length:] & response_mask)
                        img_entropy_loss = verl_F.masked_mean(entropy, seq_img_mask[..., -response_length:] & response_mask)
                    else:
                        raise ValueError(f"Invalid update_mode: {update_mode}. Must be 'text', 'image', or 'all'")
                    
                    if not self.use_adaptive_entropy_coeff:
                        entropy_coeff = self.config.entropy_coeff
                    else:
                        # update the adaptive entropy coeff
                        # entropy_coeff = -self.adaptive_entropy_coeff.alpha.detach().item()
                        # update the adaptive entropy coeff
                        # self.adaptive_entropy_coeff.update(entropy=entropy_loss.detach())
                        text_entropy_coeff = -self.text_adaptive_entropy_coeff.alpha.detach().item()
                        img_entropy_coeff = -self.img_adaptive_entropy_coeff.alpha.detach().item()
                        self.text_adaptive_entropy_coeff.update(entropy=text_entropy_loss.detach())
                        self.img_adaptive_entropy_coeff.update(entropy=img_entropy_loss.detach())
                        metrics['actor/text_entropy_coeff'] = text_entropy_coeff
                        metrics['actor/img_entropy_coeff'] = img_entropy_coeff

                    # compute policy loss
                    if not self.use_adaptive_entropy_coeff:
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss - text_entropy_loss * text_entropy_coeff - img_entropy_loss * img_entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=self.config.kl_loss_type)
                        kl_loss = masked_mean(kld, response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                        loss = torch.nan_to_num(loss, nan=0.0)
                    loss.backward()

                    data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/text_entropy_loss': text_entropy_loss.detach().item(),
                        'actor/img_entropy_loss': img_entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

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


class DataParallelDiffusionPPOActor(BasePPOActor):
    """
    PPO Actor for Diffusion models (like OmniGen2).
    Unlike the standard DataParallelPPOActor which handles autoregressive language models,
    this actor handles diffusion models by working with stored latents from rollout
    and recomputing log probabilities for specific timesteps during training.
    """

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        
        print(f'DiffusionActor initialized')

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for diffusion models using stored latents and pipeline's compute_log_prob.
        
        Args:
            micro_batch: Dictionary containing stored latents, timesteps, and embeddings for a single timestep
            temperature: Not used in diffusion, kept for interface compatibility
            
        Returns: 
            entropy: # (bs,) - entropy computed from log probabilities 
            log_probs: # (bs, 1) - log probabilities for the single timestep
        """
        # Use the pipeline's compute_log_prob method directly
        log_probs = self.actor_module.compute_log_prob(
            latents=micro_batch['latents'],
            next_latents=micro_batch['next_latents'],
            timesteps=micro_batch['timesteps'],
            prompt_embeds=micro_batch['prompt_embeds'],
            prompt_attention_mask=micro_batch['prompt_attention_mask'],
            negative_prompt_embeds=micro_batch['negative_prompt_embeds'],
            negative_prompt_attention_mask=micro_batch['negative_prompt_attention_mask'],
            ref_latents=micro_batch.get('ref_latents', None),
            noise_level=self.config.get('noise_level', 0.3),
            num_inference_steps=self.config.get('num_inference_steps', 50),
            # one line in below three is critical for correct log_prob computation
            text_guidance_scale=self.config.get('text_guidance_scale', getattr(self.actor_module, '_text_guidance_scale', 5.0)),
            image_guidance_scale=self.config.get('image_guidance_scale', getattr(self.actor_module, '_image_guidance_scale', 2.0)),
            cfg_range=self.config.get('cfg_range', getattr(self.actor_module, '_cfg_range', (0.0, 1.0))),
        )
        
        # Compute entropy from log probabilities
        # For single timestep, log_probs has shape (batch_size, 1)
        entropy = -log_probs.squeeze(1)  # (batch_size,)
        
        return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        # Debug: Check gradient status for all transformer parameters
        # print("=== Checking transformer parameter gradients ===")
        # total_params = 0
        # params_with_grad = 0
        # params_without_grad = 0
        
        # for name, param in self.actor_module.transformer.named_parameters():
        #     total_params += 1
        #     if param.grad is not None:
        #         params_with_grad += 1
        #         grad_norm = param.grad.norm().item()
        #         print(f"✓ {name}: grad_norm={grad_norm:.6f}, shape={param.shape}")
        #     else:
        #         params_without_grad += 1
        #         print(f"✗ {name}: NO GRADIENT, shape={param.shape}")
        
        # print(f"Summary: {params_with_grad}/{total_params} parameters have gradients")
        # print(f"Parameters without gradients: {params_without_grad}")
        # print("=" * 50)

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.transformer.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """
        Compute log probabilities using stored latents from rollout.
        This is called by RayPPOTrainer.fit() to derive both log_prob and ref_log_prob.
        """
        # Set to eval
        if hasattr(self.actor_module, 'eval'):
            self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']

        # Select keys needed for diffusion (stored latents data)
        required_keys = [
            'latents', 'next_latents', 'timesteps',
            # Embeddings are now padded tensors in regular batch (not non_tensor_batch)
            # 'prompt_embeds', 'prompt_attention_mask', 
            'negative_prompt_embeds', 'negative_prompt_attention_mask'
        ]
        optional_keys = ['ref_latents']
        
        # Only freqs_cis remains in non_tensor_batch (if present)
        required_non_tensor_keys = ['prompt_embeds', 'prompt_attention_mask']
        optional_non_tensor_keys = ['freqs_cis']
        
        # Check if all required keys are present
        available_keys = set(data.batch.keys())
        available_non_tensor_keys = set(data.non_tensor_batch.keys())
        
        missing_required_keys = [key for key in required_keys if key not in available_keys]
        missing_required_non_tensor_keys = [key for key in required_non_tensor_keys if key not in available_non_tensor_keys]
        
        if missing_required_keys or missing_required_non_tensor_keys:
            raise ValueError(
                f"Missing required diffusion data keys. "
                f"Missing tensor keys: {missing_required_keys}. "
                f"Missing non-tensor keys: {missing_required_non_tensor_keys}. "
                f"Available tensor keys: {list(available_keys)}. "
                f"Available non-tensor keys: {list(available_non_tensor_keys)}. "
                f"Make sure the rollout pipeline is returning diffusion_data correctly."
            )
        
        # Include optional keys that are present
        select_keys = required_keys + [key for key in optional_keys if key in available_keys]
        select_non_tensor_keys = required_non_tensor_keys + [key for key in optional_non_tensor_keys if key in available_non_tensor_keys]
        
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs'] + select_non_tensor_keys
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        else:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            micro_batches = data.select(select_keys, select_non_tensor_keys).chunk(num_micro_batches)
        
        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                processed_non_tensor_batch = self._process_non_tensor_data(micro_batch.non_tensor_batch, micro_batch.batch)
                micro_batch = {**micro_batch.batch, **processed_non_tensor_batch}
            with torch.no_grad():  # no need for grads since it's only called by old_log_prob and ref_log_prob
                _, log_probs = self._forward_micro_batch(micro_batch=micro_batch, temperature=temperature)
                log_probs_lst.append(log_probs)
        
        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs
    
    def _process_non_tensor_data(self, non_tensor_data, batch_data, device=None):
        """ non tensor data is np.array(dtype=object), transform to proper dtype and device"""
        if device is None:
            device = batch_data['latents'].device
        for key, value in non_tensor_data.items():
            if 'prompt_embeds' in key:
                dtype = batch_data['negative_prompt_embeds'].dtype
                converted = _convert_to_tensor(value, 
                                            device=device,
                                            dtype=dtype)
            elif 'prompt_attention_mask' in key:  # dtype different from prompt embeds
                dtype = batch_data['negative_prompt_attention_mask'].dtype
                converted = _convert_to_tensor(value, 
                                            device=device,
                                            dtype=dtype)
            else:
                dtype = batch_data['latents'].dtype
                converted = _convert_to_tensor(value, 
                                            device=device,
                                            dtype=dtype)
            non_tensor_data[key] = converted
        return non_tensor_data
        
    def update_policy(self, data: DataProto):
        """
        Update policy for diffusion models using stored latents.
        """
        # Make sure we are in training mode
        if hasattr(self.actor_module, 'train'):
            self.actor_module.train()

        temperature = data.meta_info['temperature']

        # Select keys for diffusion training
        required_keys = [
            'latents', 'next_latents', 'timesteps', 'old_log_probs', 'advantages',
            # Embeddings are now padded tensors in regular batch (not non_tensor_batch)
            # 'prompt_embeds', 'prompt_attention_mask', 
            'negative_prompt_embeds', 'negative_prompt_attention_mask'
        ]
        optional_keys = ['ref_latents']
        
        # Only freqs_cis remains in non_tensor_batch (if present)
        required_non_tensor_keys = ['prompt_embeds', 'prompt_attention_mask']
        optional_non_tensor_keys = ['freqs_cis']
        
        if self.config.use_kl_loss:
            required_keys.append('ref_log_prob')
        
        # Include optional keys that are present
        available_keys = set(data.batch.keys())
        available_non_tensor_keys = set(data.non_tensor_batch.keys())
        select_keys = required_keys + [key for key in optional_keys if key in available_keys]
        select_non_tensor_keys = required_non_tensor_keys + [key for key in optional_non_tensor_keys if key in available_non_tensor_keys]
        
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs', 'uid'] + select_non_tensor_keys
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['uid'] + select_non_tensor_keys
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)

        metrics = {}
        
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # Split batch into micro_batches
                mini_batch = data
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        processed_non_tensor_batch = self._process_non_tensor_data(data.non_tensor_batch, data.batch, device=torch.cuda.current_device())
                        data = {**data.batch.to(torch.cuda.current_device()), **processed_non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())
                    
                    old_log_prob = data['old_log_probs']  # (batch_size, num_steps)
                    advantages = data['advantages']  # (batch_size, num_steps) or (batch_size,)
                    uids = data['uid']
                    # Get dimensions
                    batch_size, num_steps = old_log_prob.shape
                    
                    # Handle advantage shapes for diffusion
                    if advantages.dim() == 1 and old_log_prob.dim() == 2:
                        # Expand advantages to match log_prob shape
                        advantages = advantages.unsqueeze(1).expand_as(old_log_prob)
                    elif advantages.dim() == 2 and old_log_prob.dim() == 2:
                        # Both have the same shape, use as is
                        pass

                    clip_ratio = self.config.clip_ratio
                    
                    # Initialize accumulators for this micro batch
                    total_pg_loss = 0.0
                    total_entropy_loss = 0.0
                    total_pg_clipfrac = 0.0
                    total_ppo_kl = 0.0
                    total_kl_loss = 0.0 if self.config.use_kl_loss else None
                    
                    # Loop over timesteps to save memory
                    log_probs_lst = []

                    effective_num_steps = num_steps//2+1
                    # for step_idx in range(0, num_steps, 2):  # Take every other step to speed up
                    for step_idx in range(0, effective_num_steps):  # Take first half of steps to speed up
                        # Extract data for current timestep
                        current_latents = data['latents'][:, step_idx]  # (batch_size, ...)
                        current_next_latents = data['next_latents'][:, step_idx]  # (batch_size, ...)
                        current_timesteps = data['timesteps'][:, step_idx]  # (batch_size,)
                        current_old_log_prob = old_log_prob[:, step_idx]  # (batch_size,)
                        current_advantages = advantages[:, step_idx]  # (batch_size,)
                        
                        # Prepare data for single timestep forward pass
                        timestep_data = {
                            'latents': current_latents.unsqueeze(1),  # (batch_size, 1, ...)
                            'next_latents': current_next_latents.unsqueeze(1),  # (batch_size, 1, ...)
                            'timesteps': current_timesteps.unsqueeze(1),  # (batch_size, 1)
                            'prompt_embeds': data['prompt_embeds'],
                            'prompt_attention_mask': data['prompt_attention_mask'],
                            'negative_prompt_embeds': data['negative_prompt_embeds'],
                            'negative_prompt_attention_mask': data['negative_prompt_attention_mask'],
                        }
                        
                        # Add optional keys
                        if 'ref_latents' in data:
                            timestep_data['ref_latents'] = data['ref_latents']
                        if 'freqs_cis' in data:
                            timestep_data['freqs_cis'] = data['freqs_cis']
                        
                        # Forward pass for single timestep
                        entropy, log_prob = self._forward_micro_batch(micro_batch=timestep_data, temperature=temperature)
                        log_probs_lst.append(log_prob)

                        # log_prob should be (batch_size, 1)
                        entropy = entropy  # entropy is already (batch_size,)
                        
                        # Create response mask for current timestep (all valid)
                        response_mask = torch.ones_like(log_prob, dtype=torch.bool)

                        # Compute policy loss for current timestep
                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                            old_log_prob=current_old_log_prob.unsqueeze(1),
                            log_prob=log_prob,
                            advantages=current_advantages.unsqueeze(1),
                            eos_mask=response_mask,
                            cliprange=clip_ratio,
                            uids=uids,
                            algo_name=self.config.algo_name
                        )
                        
                        # Compute entropy loss for current timestep
                        entropy_loss = entropy.mean()
                        
                        # Compute policy loss for current timestep
                        entropy_coeff = self.config.entropy_coeff
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                        
                        # Handle KL loss if enabled
                        if self.config.use_kl_loss:
                            current_ref_log_prob = data['ref_log_prob'][:, step_idx]  # (batch_size,)
                            kld = core_algos.kl_penalty(logprob=log_prob,
                                                        ref_logprob=current_ref_log_prob.unsqueeze(1),
                                                        kl_penalty=self.config.kl_loss_type)
                            kl_loss = masked_mean(kld, response_mask)
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            total_kl_loss += kl_loss.detach().item()
                        
                        # Check if gradients changed after backward
                        # if step_idx > 0:
                        #     grad_norm_before = torch.sqrt(sum(p.grad.norm()**2 for p in self.actor_module.transformer.parameters() if p.grad is not None)).item()
                        # else:
                        #     grad_norm_before = 0.0

                        # Scale loss for gradient accumulation across both micro batches and timesteps
                        # loss = policy_loss / (self.gradient_accumulation)  # flow GRPO is not dividing loss by effective_num_steps explicitly, but handled by accelerator
                        loss = policy_loss / (self.gradient_accumulation * effective_num_steps)
                        loss = torch.nan_to_num(loss, nan=0.0)
                        loss.backward()

                        # Check if gradients changed after backward
                        # grad_norm_after = torch.sqrt(sum(p.grad.norm()**2 for p in self.actor_module.transformer.parameters() if p.grad is not None)).item()
                        # diff = grad_norm_after - grad_norm_before
                        # print(f'Step {step_idx}: Gradient norm diff = {diff:.6f}')
                        
                        # Accumulate losses for logging
                        total_pg_loss += pg_loss.detach().item()
                        total_entropy_loss += entropy_loss.detach().item()
                        total_pg_clipfrac += pg_clipfrac.detach().item()
                        total_ppo_kl += ppo_kl.detach().item()

                        
                    
                    # print('log_probs_lst', log_probs_lst)
                    # Average the accumulated losses over timesteps
                    avg_pg_loss = total_pg_loss / num_steps
                    avg_entropy_loss = total_entropy_loss / num_steps
                    avg_pg_clipfrac = total_pg_clipfrac / num_steps
                    avg_ppo_kl = total_ppo_kl / num_steps
                    
                    # Prepare metrics dict
                    step_data = {
                        'actor/entropy_loss': avg_entropy_loss,
                        'actor/pg_loss': avg_pg_loss,
                        'actor/pg_clipfrac': avg_pg_clipfrac,
                        'actor/ppo_kl': avg_ppo_kl,
                    }
                    
                    if self.config.use_kl_loss:
                        avg_kl_loss = total_kl_loss / num_steps
                        step_data['actor/kl_loss'] = avg_kl_loss
                        step_data['actor/kl_coef'] = self.config.kl_loss_coef
                    
                    append_to_dict(metrics, step_data)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
                append_to_dict(metrics, data)
                
        self.actor_optimizer.zero_grad()
        return metrics
        
