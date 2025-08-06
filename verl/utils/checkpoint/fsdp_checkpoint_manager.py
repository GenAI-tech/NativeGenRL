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

import ray
import os

import warnings
from typing import Union
import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp import ShardedStateDictConfig, ShardedOptimStateDictConfig

from verl.utils.fs import copy_to_local, is_non_local

from transformers import PreTrainedTokenizer, ProcessorMixin

from .checkpoint_manager import BaseCheckpointManager


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    Supports both FSDP and non-FSDP models.

    We save 
    - sharded model states and optimizer states (for FSDP) or regular state dicts (for non-FSDP)
    - full lr_scheduler states
    - huggingface tokenizer/processor and config for ckpt merge
    """

    def __init__(self,
                 model: Union[FSDP, torch.nn.Module],
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
                 checkpoint_contents: list = ['model', 'hf_model', 'optimizer', 'extra'],
                 force_non_fsdp: bool = False,
                 **kwargs):
        """
        Args:
            model: FSDP or regular torch.nn.Module
            optimizer: torch optimizer
            lr_scheduler: torch lr scheduler
            processing_class: tokenizer or processor
            checkpoint_contents: list of contents to save
            force_non_fsdp: if True, treat model as non-FSDP even if it's wrapped
        """

        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn("`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning)
            processing_class = kwargs.pop("tokenizer")
        assert "model" in checkpoint_contents and "optimizer" in checkpoint_contents and "extra" in checkpoint_contents, f"FSDPCheckpointManager must include ['model', 'hf_model', 'optimizer', 'extra'], got {checkpoint_contents}"

        # Detect if model is FSDP wrapped
        self.is_fsdp = isinstance(model, FSDP) and not force_non_fsdp
        
        super().__init__(model,
                         optimizer,
                         lr_scheduler=lr_scheduler,
                         processing_class=processing_class,
                         checkpoint_contents=checkpoint_contents)

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        if local_path is None:
            return

        # every rank download its own checkpoint
        # Lines 87-90 - CORRECT: Differentiate FSDP vs non-FSDP
        if self.is_fsdp:
            # FSDP: Each rank loads its own shard
            remote_model_path = os.path.join(local_path, f'model_world_size_{self.world_size}_rank_{self.rank}.pt')
        else:
            # Non-FSDP: All ranks load the same model file
            remote_model_path = os.path.join(local_path, f'model_world_size_{self.world_size}.pt')

        # Optimizer and extra states: FSDP per-rank, non-FSDP shared
        if self.is_fsdp:
            remote_optim_path = os.path.join(local_path, f'optim_world_size_{self.world_size}_rank_{self.rank}.pt')
            remote_extra_state_path = os.path.join(local_path, f'extra_state_world_size_{self.world_size}_rank_{self.rank}.pt')
        else:
            remote_optim_path = os.path.join(local_path, f'optim_world_size_{self.world_size}.pt')
            remote_extra_state_path = os.path.join(local_path, f'extra_state_world_size_{self.world_size}.pt')

        print(
            f'[rank-{self.rank}]: Loading from {remote_model_path} and {remote_optim_path} and {remote_extra_state_path}'
        )
        local_model_path = copy_to_local(remote_model_path)
        local_optim_path = copy_to_local(remote_optim_path)
        local_extra_state_path = copy_to_local(remote_extra_state_path)

        model_state_dict = torch.load(local_model_path, weights_only=False)
        optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
        extra_state_dict = torch.load(local_extra_state_path, weights_only=False)

        if del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                print(
                    f'[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored'
                )

        lr_scheduler_state_dict = extra_state_dict['lr_scheduler']

        if self.is_fsdp:
            # FSDP model loading
            state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                self.model.load_state_dict(model_state_dict)
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(optimizer_state_dict)
        else:
            # Non-FSDP model loading
            self.model.load_state_dict(model_state_dict)
            if self.optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
                
        # recover random state
        if 'rng' in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(extra_state_dict['rng'])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path
        if max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0 and len(
                self.previous_saved_paths) >= max_ckpt_to_keep:
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # every rank will save its own model and optim shard
        if self.is_fsdp:
            # FSDP model saving
            state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                    model_state_dict = self.model.state_dict()
                    if self.optimizer is not None:
                        optimizer_state_dict = self.optimizer.state_dict()
                    else:
                        optimizer_state_dict = None
        else:
            # Non-FSDP model saving
            model_state_dict = self.model.state_dict()
            if self.optimizer is not None:
                optimizer_state_dict = self.optimizer.state_dict()
            else:
                optimizer_state_dict = None
                
        if self.lr_scheduler is not None:
            lr_scheduler_state_dict = self.lr_scheduler.state_dict()
        else:
            lr_scheduler_state_dict = None

        extra_state_dict = {
            'lr_scheduler': lr_scheduler_state_dict,
            'rng': self.get_rng_state(),
        }
        # Lines 183-192 - CORRECT: Differentiate FSDP vs non-FSDP
        if self.is_fsdp:
            # FSDP: Each rank saves its own shard
            model_path = os.path.join(local_path, f'model_world_size_{self.world_size}_rank_{self.rank}.pt')
            print(f'[rank-{self.rank}]: Saving FSDP model shard to {os.path.abspath(model_path)}')
            torch.save(model_state_dict, model_path)
        else:
            # Non-FSDP: Only rank 0 saves the model (all ranks have identical copies)
            if self.rank == 0:
                model_path = os.path.join(local_path, f'model_world_size_{self.world_size}.pt')
                print(f'[rank-{self.rank}]: Saving non-FSDP model to {os.path.abspath(model_path)}')
                torch.save(model_state_dict, model_path)
            else:
                print(f'[rank-{self.rank}]: Skipping model save (non-FSDP, only rank 0 saves)')

        # Optimizer and extra states: FSDP per-rank, non-FSDP rank-0 only
        if self.is_fsdp:
            # FSDP: Each rank saves its own optimizer and extra state
            optim_path = os.path.join(local_path, f'optim_world_size_{self.world_size}_rank_{self.rank}.pt')
            extra_path = os.path.join(local_path, f'extra_state_world_size_{self.world_size}_rank_{self.rank}.pt')
            print(f'[rank-{self.rank}]: Saving FSDP optimizer to {os.path.abspath(optim_path)}')
            print(f'[rank-{self.rank}]: Saving FSDP extra_state to {os.path.abspath(extra_path)}')
            torch.save(optimizer_state_dict, optim_path)
            torch.save(extra_state_dict, extra_path)
        else:
            # Non-FSDP: Only rank 0 saves optimizer and extra state
            if self.rank == 0:
                optim_path = os.path.join(local_path, f'optim_world_size_{self.world_size}.pt')
                extra_path = os.path.join(local_path, f'extra_state_world_size_{self.world_size}.pt')
                print(f'[rank-{self.rank}]: Saving non-FSDP optimizer to {os.path.abspath(optim_path)}')
                print(f'[rank-{self.rank}]: Saving non-FSDP extra_state to {os.path.abspath(extra_path)}')
                torch.save(optimizer_state_dict, optim_path)
                torch.save(extra_state_dict, extra_path)
            else:
                print(f'[rank-{self.rank}]: Skipping optimizer/extra_state save (non-FSDP, only rank 0 saves)')

        if "hf_model" in self.checkpoint_contents:
            # wait for everyone to dump to local
            torch.distributed.barrier()

            if self.rank == 0:
                hf_local_path = os.path.join(local_path, 'huggingface')
                os.makedirs(hf_local_path, exist_ok=True)
                
                # Handle different types of config objects
                try:
                    if self.is_fsdp:
                        config = self.model._fsdp_wrapped_module.config
                    else:
                        config = self.model.config
                    
                    # Check if config has save_pretrained method (HuggingFace config)
                    if hasattr(config, 'save_pretrained') and callable(getattr(config, 'save_pretrained')):
                        config.save_pretrained(hf_local_path)
                    else:
                        # Handle FrozenDict or other config types
                        print(f"Warning: Config object of type {type(config)} doesn't have save_pretrained method. Skipping config save.")
                        if hasattr(config, '__iter__') or hasattr(config, '__dict__'):
                            # Try to convert to dict and save as JSON
                            import json
                            if hasattr(config, '__iter__') and not isinstance(config, str):
                                config_dict = dict(config)  # For FrozenDict
                            else:
                                config_dict = config.__dict__  # For other objects
                            
                            with open(os.path.join(hf_local_path, 'config.json'), 'w') as f:
                                json.dump(config_dict, f, indent=2, default=str)
                            print(f"Saved config as JSON to {hf_local_path}/config.json")
                        else:
                            print(f"Unable to save config of type {type(config)}")
                            
                except AttributeError as e:
                    print(f"Warning: Could not access model config: {e}. Skipping config save.")
                
                # Save tokenizer/processor
                try:
                    self.processing_class.save_pretrained(hf_local_path)
                except Exception as e:
                    print(f"Warning: Could not save processing class: {e}")

            torch.distributed.barrier()
            
            # This should ALWAYS execute, not just when saving hf_model
            self.previous_saved_paths.append(local_path)
