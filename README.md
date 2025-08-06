# NativeGenRL

A **Verl** reinforcement learning framework for training native generation models, i.e., text transformer + image diffusion, (like OmniGen2) for improved image generation quality. This project implements GRPO with reward model (QwenVL reward, HPSv2) feedback to fine-tune models.

This is probably the **first Verl repo that supports optimization for diffusion models**.

The target of this repo to support Verl for both discrete text tokens and continuous image tokens in native generation models.

Compared with:
- **ReasonGen-R1 & T2I-R1**: which optimizes Janus-pro as a purely transformer-based native generation model, images are represented with discrete tokens, therefore fully compatible with Verl for LLM transformer optimization;

- **BLIP3o-NEXT**: although it has both text transformer + image diffusion, the RL optimizes only discrete image tokens as additional output of transformer, no gradient for diffusion process, therefore similar to above purely transformer model optimization.

Ours support optimizing continuous image tokens directly within Verl framework.

## Status (WIP)

## TODO
- [ ] improve pipeline efficiency, current too slow
- [ ] support optimizing text tokens or hidden states
- [ ] tune to work for more rewards

## 🚀 Features

- **Distributed RL Training**: Built on VERL framework for scalable multi-GPU training
- **OmniGen2 Integration**: Native support for OmniGen2 diffusion models
- **Reward Model Support**: Integration with Qwen2.5-VL and other vision-language models HPSv2 for scoring
- **Multiple Datasets**: Support for GenEval, T2I, DPG, and custom datasets
- **Flexible Configuration**: Hydra-based configuration system for easy experimentation
- **Comprehensive Monitoring**: WandB integration with detailed metrics tracking

## 📋 Requirements

- Python 3.10+
- CUDA 12.1+ (for GPU training)
- 70GB+ GPU memory recommended
- Multi-GPU setup supported

## 🛠️ Installation

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd NativeGenRL

# Run the installation script
bash install.sh
```



## 🚦 Quick Start

### Start Training

```bash
# For SLURM clusters
sbatch scripts/rl.sh

# For local training (modify script to remove SLURM directives)
bash scripts/rl.sh
```

## 📚 Usage

### Training Configuration

The training uses Hydra configuration. Key parameters in `scripts/rl.sh`:

```bash
# Data Configuration
data.train_files="[examples/rl_prompts/geneval_train.txt]"
data.val_files="[examples/rl_prompts/geneval_val.txt]"
data.train_batch_size=32

# Model Configuration  
actor_rollout_ref.model.path=$MODEL_PATH
actor_rollout_ref.model.cfg_weight=1.0

# Training Configuration
actor_rollout_ref.actor.ppo_mini_batch_size=16
actor_rollout_ref.actor.entropy_coeff=-0.00
trainer.total_epochs=2
trainer.max_steps=1600

# Reward Model Configuration
reward_model.model.path=$RM_MODEL_PATH
reward_model.name=hpsv2
```

### Custom Reward Functions

You can implement custom reward functions:

```python
def my_reward_function(batch):
    """Custom reward function"""
    # Your reward computation here
    return reward_tensor
```

### Datasets

The framework supports several dataset formats:

- **GenEval**: Simple text prompts for generation evaluation
- **T2I**: Text-to-image datasets  
- **DPG**: Data preference generation
- **Custom**: Your own prompt files

### Model Outputs

Generated models and checkpoints are saved to:
```
NativeGenRl/ckpt/ngrl/$timestamp
├── model_weights/
├── optimizer_state/
└── training_logs/
```

Generated sample images are saved to:
```
generated_samples/
```

## 🧪 Testing

The project includes comprehensive tests:

```bash
# Run all tests
bash run_tests.sh

# Individual tests
python test_actor_basic.py          # Basic actor functionality
python test_image_generation.py     # Image generation pipeline  
python test_omnigen_pipeline.py     # OmniGen2 pipeline tests
```

## 📊 Monitoring

Training metrics are automatically logged to WandB:

- **Reward Metrics**: `critic/score/mean`, `critic/rewards/mean`
- **Training Metrics**: `actor/pg_loss`, `actor/entropy_loss` 
- **Performance**: `actor/grad_norm`, response lengths
- **Validation**: Per-dataset validation scores

Access your runs at: https://wandb.ai/your-username/verl_ngrl


## 📖 Examples

### Validation Only

```bash
# Run validation without training
python3 -m verl.trainer.image_generation_rl \
    +trainer.val_only=True \
    data.val_files="[examples/rl_prompts/geneval_val.txt]"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0. See LICENSE file for details.

## 🙏 Acknowledgments

- **OmniGen2**: Advanced diffusion model for image generation
- **VERL**: Distributed reinforcement learning framework
- **ReasonGen-R1**: RL framework for Janus-pro model
- **Qwen2.5-VL**: Vision-language model for reward computation

## 📞 Support

For questions and issues:
- Open an issue on GitHub
- Check the troubleshooting section
- Review test files for usage examples

---

**Happy training! 🎨✨**
