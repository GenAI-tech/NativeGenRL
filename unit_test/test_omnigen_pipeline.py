#!/usr/bin/env python3
"""
Simple test script to verify OmniGen2Pipeline loading.
"""

import torch
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_omnigen_pipeline():
    print("🧪 Testing OmniGen2Pipeline loading...")
    
    try:
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        print("✅ OmniGen2Pipeline imported successfully")
        
        # Test loading with HuggingFace identifier
        model_path = 'OmniGen2/OmniGen2'
        print(f"📡 Loading pipeline from: {model_path}")
        
        pipeline = OmniGen2Pipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        print("✅ Pipeline loaded successfully!")
        print(f"   - Processor type: {type(pipeline.processor)}")
        print(f"   - Tokenizer type: {type(pipeline.processor.tokenizer)}")
        print(f"   - Transformer type: {type(pipeline.transformer)}")
        print(f"   - VAE type: {type(pipeline.vae)}")
        print(f"   - Scheduler type: {type(pipeline.scheduler)}")
        
        # Test a simple generation
        test_prompt = "A beautiful sunset over the ocean"
        print(f"\n🎨 Testing image generation with prompt: '{test_prompt}'")
        
        with torch.no_grad():
            output = pipeline(
                prompt=[test_prompt],
                height=512,
                width=512,
                num_inference_steps=20,  # Fewer steps for quick test
                text_guidance_scale=3.0,
                return_dict=False
            )
            
        print(f"✅ Image generation successful!")
        print(f"   - Generated {len(output)} images")
        print(f"   - Image size: {output[0].size}")
        
        # Save test image
        output[0].save("test_generation.png")
        print("💾 Saved test image as 'test_generation.png'")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure OmniGen2 is properly installed")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🔬 OmniGen2Pipeline Test")
    print("=" * 50)
    
    success = test_omnigen_pipeline()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1) 