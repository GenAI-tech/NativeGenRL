
import json
from transformers import GenerationConfig
import os

def load_generation_config_from_subfolder(model_path, subfolder, trust_remote_code=False):
    """Load GenerationConfig from a subfolder within a model directory or HF model ID."""
    # Try local file first
    config_file = os.path.join(model_path, subfolder, "generation_config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                return GenerationConfig(**json.load(f))
        except Exception:
            pass
    
    # Try HF hub with subfolder
    try:
        return GenerationConfig.from_pretrained(model_path, subfolder=subfolder, trust_remote_code=trust_remote_code)
    except Exception:
        pass
    
    # Try main model path
    try:
        return GenerationConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except Exception:
        pass
    
    # Return default if all fails
    return GenerationConfig()

# Test
result = load_generation_config_from_subfolder('OmniGen2/OmniGen2', 'mllm', trust_remote_code=True)
print(f"Result: {result}")
print(f"Config keys: {list(result.to_dict().keys())}") 