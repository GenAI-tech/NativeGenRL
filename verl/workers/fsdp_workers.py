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
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings
import psutil
import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, open_dict, OmegaConf
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_model_to_cpu, load_fsdp_optimizer, \
    load_fsdp_model_to_gpu
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from mathruler.grader import extract_boxed_content
from qwen_vl_utils import process_vision_info
import PIL
from io import BytesIO
import base64
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, MixedPrecision

from codetiming import Timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO')) # export VERL_PPO_LOGGING_LEVEL=INFO to show in .err


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    else:
        device_mesh = init_device_mesh('cuda',
                                       mesh_shape=(world_size // fsdp_size, fsdp_size),
                                       mesh_dim_names=['ddp', 'fsdp'])
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group()

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= (self.device_mesh.size() // self.ulysses_sequence_parallel_size)
            assert self.config.actor.ppo_mini_batch_size > 0, f'ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after normalization'
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (self.device_mesh.size() //
                                                            self.ulysses_sequence_parallel_size)
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, \
                    f'normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}'
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, \
                    f'normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}'

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (self.device_mesh.size() //
                                                               self.ulysses_sequence_parallel_size)
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= (self.device_mesh.size() //
                                                           self.ulysses_sequence_parallel_size)
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               use_remove_padding=False,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False,
                               use_liger=False,
                               role='actor'):
        from verl.utils.model import print_model_size, update_model_config, get_generation_config, load_generation_config_from_subfolder
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForVision2Seq
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, CPUOffload
        from torch import optim

        assert role in ['actor', 'ref']

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        
        # Check if this is OmniGen2 model
        if 'omni' in model_path.lower(): 
            # Load OmniGen2Pipeline directly using from_pretrained
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2_chat import OmniGen2ChatPipeline
            from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
            from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
            from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

            # Determine which pipeline to use based on configuration
            use_chat_pipeline = self.config.get('use_chat_pipeline', False)
            
            if use_chat_pipeline:
                # Load OmniGen2ChatPipeline for multimodal chat capabilities
                pipeline = OmniGen2ChatPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=trust_remote_code
                )
            else:
                # Load standard OmniGen2Pipeline for image generation only
                pipeline = OmniGen2Pipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=trust_remote_code
                )
            if not self.config.rollout.get('cfg_enabled', True):
                pipeline.disable_cfg()  # disable cfg for rollout and log prob compute can reduce computation 

            scheduler = self.config.rollout.get('scheduler', 'euler')

            if scheduler == 'euler':
                pipeline.scheduler = FlowMatchEulerDiscreteScheduler()
            elif scheduler == 'dpmsolver++':
                pipeline.scheduler = DPMSolverMultistepScheduler(
                    algorithm_type="dpmsolver++",
                    solver_type="midpoint",
                    solver_order=2,
                    prediction_type="flow_prediction",
                )

            pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            )
            
            # Ensure all pipeline components are on the same device
            device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
            pipeline = pipeline.to(device)
            if hasattr(pipeline, 'transformer'):
                pipeline.transformer = pipeline.transformer.to(device)
            if hasattr(pipeline, 'vae'):
                pipeline.vae = pipeline.vae.to(device)
            if hasattr(pipeline, 'text_encoder'):
                pipeline.text_encoder = pipeline.text_encoder.to(device)
                    
            # Set tokenizer and processor from the pipeline
            self.tokenizer = pipeline.processor.tokenizer
            self.processor = pipeline.processor
        
            # Set generation config
            # OSError: OmniGen2/OmniGen2 does not appear to have a file named generation_config.json. Checkout 'https://huggingface.co/OmniGen2/OmniGen2/tree/main'for available files.
            # it's under the subfolder mllm
            self.generation_config = load_generation_config_from_subfolder(model_path, 'mllm', trust_remote_code=trust_remote_code)
            print(f"Generation config: {self.generation_config}")

            if self.generation_config is None:
                from transformers import GenerationConfig
                self.generation_config = GenerationConfig()
            self.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.generation_config.eos_token_id = self.tokenizer.eos_token_id
            self.generation_config.text_guidance_scale = self.config.rollout.get('text_guidance_scale', 1.0)
            self.generation_config.image_guidance_scale = self.config.rollout.get('image_guidance_scale', 1.0)

            
            # Update rollout config
            OmegaConf.set_struct(self.config.rollout, True)
            # with open_dict(self.config.rollout):
            #     self.config.rollout.cfg_weight = self.config.model.get('cfg_weight', 1.0)
                
        else:
            # For non-OmniGen2 models, load tokenizer and processor normally
            self.tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
            self.processor = hf_processor(model_path, trust_remote_code=trust_remote_code)
            self.generation_config = get_generation_config(model_path, trust_remote_code=trust_remote_code)
            
        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        
        # TODO omni also need actor model config
        # For OmniGen2, we need to return a dummy config for compatibility
        if 'omni' in model_path.lower():
            actor_model_config = type('Config', (), {'tie_word_embeddings': False})()
        else:
            actor_model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')
        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings,
                                                    mesh=self.device_mesh)
            
        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)


        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if 'omni' in model_path.lower():
                actor_module = pipeline.transformer
                # to bf16
                actor_module.to(torch.bfloat16)
                actor_module_fsdp = actor_module  # TODO no fsdp for now
            else:
                if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                    actor_module_class = AutoModelForVision2Seq
                else:
                    actor_module_class = AutoModelForCausalLM

                actor_module = actor_module_class.from_pretrained(pretrained_model_name_or_path=model_path,
                                                                torch_dtype=torch_dtype,
                                                                config=actor_model_config,
                                                                attn_implementation='flash_attention_2',
                                                                trust_remote_code=trust_remote_code)

                if use_remove_padding or self.ulysses_sequence_parallel_size > 1:
                    from verl.models.transformers.monkey_patch import apply_monkey_patch
                    apply_monkey_patch(model=actor_module)

                # Apply Liger kernel to the model if use_liger is set to True
                if use_liger:
                    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                    _apply_liger_kernel_to_instance(model=actor_module)

                # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
                actor_module.to(torch_dtype)

                if enable_gradient_checkpointing:
                    if 'omni' not in model_path.lower():   # TODO 'OmniGen2Transformer2DModel' object has no attribute 'gradient_checkpointing_enable'
                        actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                
                torch.distributed.barrier()

                if self.rank == 0:
                    print_model_size(actor_module)
                    # print(list(actor_module.named_modules()))

                log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

                # We wrap FSDP for rollout as well
                mixed_precision_config = fsdp_config.get('mixed_precision', None)
                if mixed_precision_config is not None:
                    param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
                    reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
                    buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
                else:
                    param_dtype = torch.bfloat16
                    reduce_dtype = torch.float32
                    buffer_dtype = torch.float32

                mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

                # For OmniGen2 models, use size-based auto wrap policy with appropriate min_num_params
                if 'omni' in model_path.lower():
                    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
                    from functools import partial
                    auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1e6)  # 1M parameters minimum
                else:
                    auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

                # if self._is_rollout and self.config.rollout.name == 'hf':
                    # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
                    # auto_wrap_policy = None

                print(f'wrap_policy: {auto_wrap_policy}')

                fsdp_mesh = self.device_mesh
                sharding_strategy = get_sharding_strategy(fsdp_mesh)

                # TODO: add transformer policy
                # We force reference policy to use CPUOffload to save memory.
                # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
                # for FSDP2 this should work: CPUOffload+grad accumulation
                cpu_offload = None if role == 'actor' else CPUOffload(offload_params=True)
                actor_module_fsdp = FSDP(
                    actor_module,
                    cpu_offload=cpu_offload,
                    param_init_fn=init_fn,
                    use_orig_params=True,
                    auto_wrap_policy=auto_wrap_policy,
                    device_id=torch.cuda.current_device(),
                    sharding_strategy=sharding_strategy,  # zero3
                    mixed_precision=mixed_precision,
                    sync_module_states=True,
                    device_mesh=self.device_mesh,
                    forward_prefetch=False)

                log_gpu_memory_usage('After Actor FSDP init', logger=logger)

                def print_wrapped(mod, prefix=""):
                    for name, child in mod.named_children():
                        cls = type(child).__name__
                        print(f"{prefix}{name}: {cls}")
                        print_wrapped(child, prefix + "  ")

                if self.rank == 0:
                    print_wrapped(actor_module_fsdp)

        # TODO: add more optimizer args into config
        if role == 'actor' and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            optimized_params = actor_module_fsdp.parameters()
            actor_optimizer = optim.AdamW(optimized_params,
                                          lr=optim_config.lr,
                                        #   betas=optim_config.get('betas', (0.9, 0.999)),
                                          betas=optim_config.get('betas', (0.9, 0.95)), # llama param, larger grad for earlier layers
                                          weight_decay=optim_config.get('weight_decay', 1e-2))

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps = int(optim_config.get('lr_warmup_steps', -1))
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)
        if 'omni' in model_path.lower():
            # pipeline.transformer = actor_module_fsdp  # TODO ?
            forward_module = pipeline
        else:
            forward_module = actor_module_fsdp

        return forward_module, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}'
        rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp'])
        rollout_name = self.config.rollout.name
        if rollout_name == 'hf':
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager import BaseShardingManager
            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?

        elif rollout_name == 'vllm':
            from verl.workers.rollout.vllm_rollout import vLLMRollout, vllm_mode
            from verl.workers.sharding_manager import FSDPVLLMShardingManager
            log_gpu_memory_usage(f'Before building {rollout_name} rollout', logger=None)
            local_path = copy_to_local(self.config.model.path)
            if vllm_mode == 'customized':
                rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
                                      config=self.config.rollout,
                                      tokenizer=self.tokenizer,
                                      model_hf_config=self.actor_model_config)
            elif vllm_mode == 'spmd':
                rollout = vLLMRollout(model_path=local_path,
                                      config=self.config.rollout,
                                      tokenizer=self.tokenizer,
                                      model_hf_config=self.actor_model_config,
                                      device_mesh=rollout_device_mesh)
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")
            log_gpu_memory_usage(f'After building {rollout_name} rollout', logger=None)
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = 'dummy_hf'
            rollout_sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
                                                               inference_engine=rollout.inference_engine,
                                                               model_config=self.actor_model_config,
                                                               full_params='hf' in self.config.rollout.load_format,
                                                               device_mesh=rollout_device_mesh)
            log_gpu_memory_usage('After building sharding manager', logger=None)

        elif rollout_name == 'sglang':
            from verl.workers.rollout.sglang_rollout import SGLangRollout
            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to SGLang's model_runner would check CUDA device capability.
            # However, due to veRL's setting, the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager
            log_gpu_memory_usage(f'Before building {rollout_name} rollout', logger=None)
            rollout = SGLangRollout(actor_module=self.config.model.path,
                                    config=self.config.rollout,
                                    tokenizer=self.tokenizer,
                                    model_hf_config=self.actor_model_config)
            log_gpu_memory_usage(f'After building {rollout_name} rollout', logger=None)

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = 'dummy_hf'
            rollout_sharding_manager = FSDPSGLangShardingManager(module=self.actor_module_fsdp,
                                                                 inference_engine=rollout.inference_engine,
                                                                 model_config=self.actor_model_config,
                                                                 full_params='hf' in self.config.rollout.load_format,
                                                                 device_mesh=rollout_device_mesh)
            log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor, DataParallelDiffusionPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        use_remove_padding = self.config.model.get('use_remove_padding', False)
        detach_uncond = self.config.model.get('detach_uncond', False)
        text_guidance_scale = self.config.rollout.get('text_guidance_scale', 1.0)
        image_guidance_scale = self.config.rollout.get('image_guidance_scale', 1.0)
        

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=self.config.model.get('trust_remote_code', False),
                use_liger=self.config.model.get('use_liger', False),
                role='actor')

            # get the original unwrapped module
            # For OmniGen2, the module is not FSDP wrapped, so we use it directly
            if hasattr(self.actor_module_fsdp, '_fsdp_wrapped_module'):
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module
            else:
                self.actor_module = self.actor_module_fsdp  # Omnigen2

            if self._is_offload_optimizer and isinstance(self.actor_module_fsdp, FSDP):
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.text_guidance_scale = text_guidance_scale
                self.config.actor.image_guidance_scale = image_guidance_scale
                self.config.actor.detach_uncond = detach_uncond
                self.config.actor.bos_token_id = self.tokenizer.bos_token_id
                self.config.actor.pad_token_id = self.tokenizer.pad_token_id
                # TODO the image start token id is not set for OmniGen2
                # Handle image_start_token_id based on model type
                if hasattr(self.processor, 'image_start_tag'):
                    self.config.actor.image_start_token_id = self.tokenizer.encode(self.processor.image_start_tag)[-1]
                else:
                    # For OmniGen2, use a default image token id
                    self.config.actor.image_start_token_id = self.get_token_id('<|img|>')

                print(f"actor config: {self.config.actor.image_start_token_id}", self.tokenizer.pad_token_id)
                
            # self.actor = DataParallelPPOActor(config=self.config.actor,
            #                                   actor_module=self.actor_module_fsdp,
            #                                   actor_optimizer=self.actor_optimizer)

            self.actor = DataParallelDiffusionPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               use_remove_padding=use_remove_padding,
                                                               trust_remote_code=self.config.model.get(
                                                                   'trust_remote_code', False),
                                                               use_liger=self.config.model.get('use_liger', False),
                                                               role='ref')[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.text_guidance_scale = text_guidance_scale
                self.config.ref.image_guidance_scale = image_guidance_scale
                self.config.ref.detach_uncond = detach_uncond
                self.config.ref.bos_token_id = self.tokenizer.bos_token_id
                self.config.ref.pad_token_id = self.tokenizer.pad_token_id
                # Handle image_start_token_id based on model type
                if hasattr(self.processor, 'image_start_tag'):
                    self.config.ref.image_start_token_id = self.tokenizer.encode(self.processor.image_start_tag)[-1]
                else:
                    # For OmniGen2, use a default image token id  
                    self.config.ref.image_start_token_id = self.get_token_id('<|img|>')
                
            # self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
            self.ref_policy = DataParallelDiffusionPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            # self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                # model=self.actor_module_fsdp,
                model=self.actor_module_fsdp.transformer,  # OmniGen2 pipeline
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=self.config.actor.checkpoint.contents,
                force_non_fsdp=not isinstance(self.actor_module_fsdp.transformer, FSDP))  # OmniGen2 pipeline is not FSDP wrapped

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        assert self._is_actor
        assert self._is_actor
        if self._is_offload_param and isinstance(self.actor_module_fsdp, FSDP):
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer and isinstance(self.actor_optimizer, FSDP):
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        log_gpu_memory_usage('Before update policy', logger=logger)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name='update_policy', logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            # estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            # metrics[
            #     'perf/mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            metrics['perf/max_memory_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
            metrics['perf/max_memory_reserved_gb'] = torch.cuda.max_memory_reserved() / (1024**3)
            metrics['perf/cpu_memory_used_gb'] = psutil.virtual_memory().used / (1024**3)

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics['actor/lr'] = lr

            log_gpu_memory_usage('After update policy', logger=logger)

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={'metrics': metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')

        if self._is_offload_param and isinstance(self.actor_module_fsdp, FSDP):
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer and isinstance(self.actor_module_fsdp, FSDP):
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        return output

    def get_token_id(self, tag):
        id = self.tokenizer.vocab.get(tag)
        return id
        
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(torch.cuda.current_device())

        assert self._is_rollout
        if self._is_offload_param and isinstance(self.actor_module_fsdp, FSDP):
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        else: 
            self.actor_module_fsdp.mllm.to(torch.cuda.current_device())  # text generation only for image generation
            self.actor_module_fsdp.vae.to(torch.cuda.current_device())

        
        meta_info = {
            'eos_token_id':
                self.generation_config.eos_token_id
                if self.generation_config is not None else self.tokenizer.eos_token_id,  # this is actually <|im_end|>
            'pad_token_id':
                self.generation_config.pad_token_id
                if self.generation_config is not None else self.tokenizer.pad_token_id,
            'image_start_token_id': self.get_token_id('<|img|>'),
            'image_end_token_id': self.get_token_id('<|im_end|>'),
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            with torch.no_grad():  # forward sequence no need for gradient

                # after parameters sync with rollout, offload actor model to CPU
                if self._is_offload_param and self.config.rollout.name=='vllm' and isinstance(self.actor_module_fsdp, FSDP):
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                if self._is_offload_optimizer and isinstance(self.actor_module_fsdp, FSDP):
                    offload_fsdp_optimizer(optimizer=self.actor_optimizer)

                log_gpu_memory_usage('After entering rollout sharding manager', logger=logger)

                prompts = self.rollout_sharding_manager.preprocess_data(prompts)
                output = self.rollout.generate_sequences(prompts=prompts)
                log_gpu_memory_usage('After rollout generation', logger=logger)

                output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to('cpu')
        if not isinstance(self.actor_module_fsdp, FSDP):
            self.actor_module_fsdp.mllm.to('cpu')
            self.actor_module_fsdp.vae.to('cpu')

        # clear kv cache
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param and isinstance(self.actor_module_fsdp, FSDP):
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        data = data.to(torch.cuda.current_device())
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info['temperature'] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            with torch.no_grad(): #  KL of old_log_prob with ref_log_prob (in ray_trainer.py) is only reward penalty, no grads
                data = self.ulysses_sharding_manager.preprocess_data(data)
                output = self.actor.compute_log_prob(data=data)
                output = DataProto.from_dict(tensors={'old_log_probs': output},
                                            meta_info={'temperature': self.config.rollout.temperature})
                output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and hasattr(self.actor.actor_module, '_handle'):
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param and isinstance(self.actor_module_fsdp, FSDP):
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        log_gpu_memory_usage('After compute_log_prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            with torch.no_grad():
                data = self.ulysses_sharding_manager.preprocess_data(data)
                output = self.ref_policy.compute_log_prob(data=data)
                output = DataProto.from_dict(tensors={'ref_log_prob': output})
                output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and hasattr(self.ref_policy.actor_module, '_handle'):
            self.ref_policy.actor_module._handle.reshard(True)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # only support save and load ckpt for actor
        assert self._is_actor
        import torch
        if self._is_offload_param and isinstance(self.actor_module_fsdp, FSDP):
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                global_step=global_step,
                                                max_ckpt_to_keep=max_ckpt_to_keep)

        torch.distributed.barrier()
        if self._is_offload_param and isinstance(self.actor_module_fsdp, FSDP):
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        if self._is_offload_param and isinstance(self.actor_module_fsdp, FSDP):
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                del_local_after_load=del_local_after_load)

        if self._is_offload_param and isinstance(self.actor_module_fsdp, FSDP):
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer and isinstance(self.actor_module_fsdp, FSDP):
            offload_fsdp_optimizer(self.actor_optimizer)


class CriticWorker(Worker):

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (torch.distributed.get_world_size() //
                                                  self.ulysses_sequence_parallel_size)
            self.config.forward_micro_batch_size //= (torch.distributed.get_world_size() //
                                                      self.ulysses_sequence_parallel_size)
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, \
                f'normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}'
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, \
                f'normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}'

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        from torch import optim

        local_path = copy_to_local(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_to_local(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))

        from omegaconf import OmegaConf
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')

        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForTokenClassification
        from torch import nn

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        critic_model_config.num_labels = 1

        init_context = get_init_weight_context_manager(use_meta_tensor=not critic_model_config.tie_word_embeddings,
                                                       mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(critic_model_config, 'classifier_dropout', 0.)
            setattr(critic_model_config, 'hidden_dropout', '0')
            critic_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            torch_dtype=torch_dtype,
                                                                            config=critic_model_config,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)

            use_remove_padding = config.model.get('use_remove_padding', False)
            if use_remove_padding or self.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch
                apply_monkey_patch(model=critic_module)

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get('enable_gradient_checkpointing', False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage('Before critic FSDP', logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        critic_module = FSDP(critic_module,
                             param_init_fn=init_fn,
                             use_orig_params=False,
                             auto_wrap_policy=auto_wrap_policy,
                             device_id=torch.cuda.current_device(),
                             sharding_strategy=sharding_strategy,
                             mixed_precision=mixed_precision,
                             sync_module_states=True,
                             forward_prefetch=False,
                             device_mesh=self.device_mesh,
                             cpu_offload=None)

        log_gpu_memory_usage('After critic FSDP', logger=None)

        critic_optimizer = optim.AdamW(critic_module.parameters(),
                                       lr=config.optim.lr,
                                       betas=config.optim.get('betas', (0.9, 0.999)),
                                       weight_decay=config.optim.get('weight_decay', 1e-2))

        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps = int(config.optim.get('lr_warmup_steps', -1))
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer,
                                                                num_warmup_steps=num_warmup_steps)

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from verl.workers.critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(config=self.config,
                                            critic_module=self.critic_module,
                                            critic_optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_contents=self.config.checkpoint.contents)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):

        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={'values': values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        # Support all hardwares
        data = data.to(torch.cuda.current_device())
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name='update_critic', logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['perf/mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics['critic/lr'] = lr

            output = DataProto(batch=None, meta_info={'metrics': metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        import torch
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                global_step=global_step,
                                                max_ckpt_to_keep=max_ckpt_to_keep)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        import torch
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                del_local_after_load=del_local_after_load)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

class QwenVLRewardModelWorker(Worker):
    """
    Qwen2.5-VL-7B-Instruct reward model worker.
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

        logger.warning("Reward Model: Qwen2.5-VL-7B-Instruct")
        self.template = self.config.template
        self.paired = self.config.paired
        self.use_remove_padding = self.config.model.get('use_remove_padding', False)

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        """
        Expand scores for diffusion models: (batch,) -> (batch, num_inference_steps)
        """
        token_level_scores = torch.zeros_like(data.batch['timesteps'], dtype=scores.dtype)  # (batch, num_inference_steps)
        token_level_scores[:, -1] = scores  # Assign scores only to last position, better match with grpo for text.  # [[0,0,0,0,r]]
        return token_level_scores

    def _build_model(self, config):
        # the following line is necessary
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path)
        self._do_switch_chat_template = True

        # input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer)
        # self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
        #                                     trust_remote_code=config.model.get('trust_remote_code', False))
        from transformers import Qwen2_5_VLProcessor
        processor = Qwen2_5_VLProcessor.from_pretrained("OmniGen2/OmniGen2", subfolder="processor")  # TODO processor or mllm_processor
        self.input_tokenizer = processor.tokenizer  # this is actor model tokenizer; OmniGen2 tokenizer is Qwen2_5_VLProcessor

        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get('trust_remote_code', False))
        self.processor = hf_processor(local_path, trust_remote_code=config.model.get('trust_remote_code', False))
        self.processor.tokenizer.padding_side = 'left'

        trust_remote_code = config.model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings,
                                                       mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(model_config, 'classifier_dropout', 0.)
            reward_module = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            config=model_config,
                                                                            torch_dtype=torch.bfloat16,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)

            if config.model.get('use_remove_padding', False) or self.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch
                apply_monkey_patch(model=reward_module)

            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=True),
            forward_prefetch=False,
            device_mesh=self.device_mesh)
        
        self.rollout_name = self.config.rollout.get('name', 'vllm')
        if self.rollout_name == 'vllm':
            from verl.workers.rollout.vllm_rollout import vLLMRollout, vllm_mode
            from verl.workers.sharding_manager import FSDPVLLMShardingManager
            infer_tp = self.config.rollout.tensor_model_parallel_size
            dp = self.world_size // infer_tp
            rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp'])
            
            self.rollout = vLLMRollout(model_path=config.model.path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=model_config,
                device_mesh=rollout_device_mesh)
            self.rollout_sharding_manager = FSDPVLLMShardingManager(module=reward_module,
                                            inference_engine=self.rollout.inference_engine,
                                            model_config=model_config,
                                            full_params='hf' in self.config.rollout.load_format,
                                            device_mesh=rollout_device_mesh)

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)
        
    def get_score_from_output(self, output_text):
        if self.paired:
            scores = torch.zeros(len(output_text)*2, device=self.device_mesh.device_type)  # two images for dpo
            for i, text in enumerate(output_text):
                better_idx = extract_boxed_content(text)
                try:
                    if better_idx in ['1', '2']:
                        better_idx = int(better_idx) - 1  # idx start from 1 in the text, minus 1 to make it start from 0
                        scores[i*2 + better_idx] = 1.0
                    else:
                        pass
                except:
                    pass
        else:
            scores = torch.zeros(len(output_text), device=self.device_mesh.device_type)
            for i, text in enumerate(output_text):
                score = extract_boxed_content(text)
                try:
                    if score in ['1', '0']:
                        scores[i] = float(score)
                except:
                    pass
        return scores

    def _forward_micro_batch(self, micro_batch):
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if self.rollout_name == 'hf':
                input_ids = micro_batch.batch['input_ids']
                attention_mask = micro_batch.batch['attention_mask']
                position_ids = micro_batch.batch['position_ids']
                generated_ids = self.reward_module.generate(
                                            input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            pixel_values=micro_batch['pixel_values'],
                                            image_grid_thw=micro_batch['image_grid_thw'].reshape(-1, 3),
                                            max_new_tokens=self.config.model.max_new_tokens,
                                            use_cache=True)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
                ]
                
            elif self.rollout_name == 'vllm':
                micro_batch.meta_info['eos_token_id'] = self.processor.tokenizer.eos_token_id
                micro_batch.meta_info['do_sample'] = False
                output = self.rollout.generate_sequences(micro_batch)
                generated_ids_trimmed = output.batch['responses']
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            # print(f'output_text: {output_text}')
            rm_score = self.get_score_from_output(output_text)
            return rm_score, output_text

    def _switch_chat_template(self, data: DataProto):
        self.max_prompt_length = self.config.model.max_prompt_length
        
        uid_group = {}
        if self.paired:
            for i in range(data.batch.batch_size[0]):
                uid = data.non_tensor_batch['uid'][i]
                if uid not in uid_group:
                    uid_group[uid] = []
                uid_group[uid].append(i)
        else:
            uid_group = {i: [i] for i in range(data.batch.batch_size[0])}  # dummy uid groups
            
        uids = []
        input_dict = {
            'input_ids': [],
            'attention_mask': [],
            'pixel_values': [],
            'image_grid_thw': [],
            'position_ids': [],
        }
        non_tensor_dict = {
            'uid': [],
            'multi_modal_data': [],
            'raw_prompt_ids': [],
        }
        
        new_rank = []
        
        for uid in uid_group:
            if self.paired:
                indices = uid_group[uid]
            else:  # uid is actually the index when not paired
                indices = [uid]
                if 'uid' in data.non_tensor_batch:
                    uid = data.non_tensor_batch['uid'][uid]
                else:
                    uid = str(uid)
                
            first_idx = indices[0]
            # extract raw prompt
            prompt = data.non_tensor_batch['raw_prompt'][first_idx]
            imgs = []
            for idx in indices:
                img = data.batch['gen_img'][idx]
                imgs.append(PIL.Image.fromarray(img.cpu().numpy()))
                new_rank.append(idx)
            chat = [{'role': 'user', 
                     'content': [{'type': 'text', 'text': self.template.format(prompt=prompt.replace("A photo of ",""))}],
                    }]
            prompt_with_chat_template = self.processor.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            uids.append(uid)
            
            imgs = [process_image(img) for img in imgs]
            multi_modal_data = {'image': imgs}
            image_inputs = self.processor.image_processor(imgs, return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            for key, value in image_inputs.items():
                input_dict[key].append(value)
                
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)

            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation='right')
            input_dict['input_ids'].append(input_ids[0])
            input_dict['attention_mask'].append(attention_mask[0])
            
            raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            non_tensor_dict['raw_prompt_ids'].append(raw_prompt_ids)
            non_tensor_dict['multi_modal_data'].append(multi_modal_data)
            non_tensor_dict['uid'].append(uid)
            
            from verl.models.transformers.qwen2_vl import get_rope_index
            input_dict['position_ids'].append(
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask[0],
                )
            )
            
        for key in input_dict:
            if isinstance(input_dict[key][0], torch.Tensor):
                input_dict[key] = torch.stack(input_dict[key], dim=0)
        
        return DataProto.from_dict(input_dict, non_tensors=non_tensor_dict), new_rank

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        import itertools
        from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
        
        load_fsdp_model_to_gpu(self.reward_module)
        # Support all hardwares
        data = data.to(torch.cuda.current_device())
        if self._do_switch_chat_template:
            rm_data, new_rank = self._switch_chat_template(data)

        # Support all hardwares
        rm_data.batch = rm_data.batch.to(torch.cuda.current_device())

        # perform forward computation
        manager = self.ulysses_sharding_manager if self.rollout_name == 'hf' else self.rollout_sharding_manager
        with manager:
            log_gpu_memory_usage('After entering rollout sharding manager', logger=logger)
            rm_data = manager.preprocess_data(data=rm_data)
            data = manager.preprocess_data(data=data)

            num_micro_batches = max(len(rm_data.batch) // self.config.micro_batch_size_per_gpu, 1)
            micro_batches = rm_data.chunk(num_micro_batches)
            output = []
            output_texts = []
            for micro_batch in micro_batches:
                rm_score, output_text = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
                output_texts.extend(output_text)
            scores = torch.cat(output, dim=0)  # (batch_size)
            log_gpu_memory_usage('After rollout generation', logger=logger)

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            
            # interleave repeat text to match the batchsize
            output_texts = [text for text in output_texts for i in range(len(token_level_scores)//len(output_texts))]
            
            output = DataProto.from_dict(tensors={'rm_scores': token_level_scores}, 
                                         non_tensors={'rm_text': output_texts})
            output = manager.postprocess_data(data=output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        self.reward_module._handle.reshard(True)
        offload_fsdp_model_to_cpu(self.reward_module)

        output = output.to('cpu')
        if self._do_switch_chat_template:
            # print(scores, new_rank)
            output.batch['rm_scores'][new_rank] = output.batch['rm_scores'][new_rank].clone()
        return output


class HPSv2RewardModelWorker(Worker):
    """
    A reward model worker that uses HPSv2 for scoring image-prompt alignment.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

        logger.warning("Reward Model: HPSv2")

    # def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
    #     batch_size = data.batch.batch_size[0]
    #     # expand as token_level_reward
    #     attention_mask = data.batch['attention_mask']
    #     position_ids = data.batch['position_ids']
    #     response_length = data.batch['responses'].shape[-1]
    #     eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
    #     token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
    #     token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

    #     # select the response part
    #     token_level_scores = token_level_scores[:, -response_length:]

    #     return token_level_scores

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        """
        Expand scores for diffusion models: (batch,) -> (batch, num_inference_steps)
        """
        # batch_size = data.batch.batch_size[0]
        shape = data.batch['timesteps'].shape
        # expand to (batch, num_inference_steps)
        # token_level_scores = scores.unsqueeze(1).expand(-1, shape[1])  # [[r,r,r,r,r]]
        # For diffusion models, assign reward only to last position
        token_level_scores = torch.zeros_like(data.batch['timesteps'], dtype=scores.dtype)  # (batch, num_inference_steps)
        token_level_scores[:, -1] = scores  # Assign scores only to last position, better match with grpo for text.  # [[0,0,0,0,r]]
        return token_level_scores

    def _build_model(self, config):
        # Initialize HPSv2
        import hpsv2
        self.hpsv2 = hpsv2
        
        # No model to build since HPSv2 is used directly
        return None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # Initialize HPSv2
        self._build_model(self.config)

    def _forward_micro_batch(self, micro_batch):
        # logger.info("micro_batch: ", micro_batch.batch.keys(), micro_batch.non_tensor_batch.keys())
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Extract images and prompts
            images = micro_batch.batch['gen_img']
            prompts = micro_batch.non_tensor_batch['raw_prompt']

            # Convert images to PIL
            pil_images = [PIL.Image.fromarray(img.cpu().numpy()) for img in images]
            
            # Calculate rewards
            rewards = []
            for img, prompt in zip(pil_images, prompts):
                reward = self.hpsv2_reward_fn(img, prompt)
                rewards.append(reward)
            
            return torch.stack(rewards), prompts

    def hpsv2_reward_fn(self, image, prompt):
        """
        HPSv2 reward function that scores an image based on its alignment with a prompt.
        
        Args:
            image: Generated image (PIL Image)
            prompt: Text prompt to score against
        
        Returns:
            torch.Tensor: Reward score
        """
        # Save image to temporary file
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            # Score image
            result = self.hpsv2.score(tmp.name, prompt, hps_version="v2.1")
            # Clean up
            os.unlink(tmp.name)
        
        return torch.tensor(result[0], device='cuda')

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            num_micro_batches = max(len(data.batch) // self.config.micro_batch_size_per_gpu, 1)
            micro_batches = data.chunk(num_micro_batches)
            output = []
            output_texts = []
            for micro_batch in micro_batches:
                rm_score, output_text = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
                output_texts.extend(output_text)
            scores = torch.cat(output, dim=0)  # (batch_size)

            token_level_scores = self._expand_to_token_level(data, scores)
            
            # interleave repeat text to match the batchsize
            output_texts = [text for text in output_texts for i in range(len(token_level_scores)//len(output_texts))]
            
            output = DataProto.from_dict(tensors={'rm_scores': token_level_scores}, 
                                         non_tensors={'rm_text': output_texts})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        return output



def create_reward_model_worker(config):
    """
    Factory function to create the appropriate reward model worker based on config.name.
    """
    reward_model_name = config.get('name', 'qwenvl').lower()
    
    if reward_model_name == 'qwenvl':
        RewardModelWorker = QwenVLRewardModelWorker
    elif reward_model_name == 'hpsv2':
        RewardModelWorker = HPSv2RewardModelWorker
    else:
        raise ValueError(f"Unknown reward model name: {reward_model_name}. "
                        f"Supported names: qwenvl, hpsv2")
    return RewardModelWorker

