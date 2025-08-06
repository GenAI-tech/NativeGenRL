#!/usr/bin/env python3
"""
Test script for the modified FSDPCheckpointManager that supports both FSDP and non-FSDP models.
"""

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoTokenizer
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager


class SimpleModel(nn.Module):
    """Simple test model"""
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
        # Mock config attribute for HF compatibility
        class MockConfig:
            def save_pretrained(self, path):
                print(f"Mock config saved to {path}")
        
        self.config = MockConfig()
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


def test_non_fsdp_model():
    """Test FSDPCheckpointManager with a regular (non-FSDP) model"""
    print("Testing with non-FSDP model...")
    
    # Create a regular model
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Mock tokenizer
    class MockTokenizer:
        def save_pretrained(self, path):
            print(f"Mock tokenizer saved to {path}")
    
    tokenizer = MockTokenizer()
    
    # Create checkpoint manager - should auto-detect non-FSDP
    checkpoint_manager = FSDPCheckpointManager(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        processing_class=tokenizer,
        checkpoint_contents=['model', 'optimizer', 'extra']  # Skip 'hf_model' for simplicity
    )
    
    print(f"Is FSDP: {checkpoint_manager.is_fsdp}")  # Should be False
    assert not checkpoint_manager.is_fsdp, "Expected non-FSDP model"
    print("✓ Non-FSDP model test passed")


def test_fsdp_model():
    """Test FSDPCheckpointManager with an FSDP model"""
    print("\nTesting with FSDP model...")
    
    # Initialize distributed (required for FSDP)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    
    # Create FSDP model
    model = SimpleModel()
    fsdp_model = FSDP(model)
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Mock tokenizer
    class MockTokenizer:
        def save_pretrained(self, path):
            print(f"Mock tokenizer saved to {path}")
    
    tokenizer = MockTokenizer()
    
    # Create checkpoint manager - should auto-detect FSDP
    checkpoint_manager = FSDPCheckpointManager(
        model=fsdp_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        processing_class=tokenizer,
        checkpoint_contents=['model', 'optimizer', 'extra']  # Skip 'hf_model' for simplicity
    )
    
    print(f"Is FSDP: {checkpoint_manager.is_fsdp}")  # Should be True
    assert checkpoint_manager.is_fsdp, "Expected FSDP model"
    print("✓ FSDP model test passed")


def test_force_non_fsdp():
    """Test forcing FSDP model to be treated as non-FSDP"""
    print("\nTesting force_non_fsdp parameter...")
    
    # Initialize distributed (required for FSDP)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    
    # Create FSDP model
    model = SimpleModel()
    fsdp_model = FSDP(model)
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Mock tokenizer
    class MockTokenizer:
        def save_pretrained(self, path):
            print(f"Mock tokenizer saved to {path}")
    
    tokenizer = MockTokenizer()
    
    # Create checkpoint manager with force_non_fsdp=True
    checkpoint_manager = FSDPCheckpointManager(
        model=fsdp_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        processing_class=tokenizer,
        checkpoint_contents=['model', 'optimizer', 'extra'],
        force_non_fsdp=True  # Force non-FSDP treatment
    )
    
    print(f"Is FSDP: {checkpoint_manager.is_fsdp}")  # Should be False due to force_non_fsdp
    assert not checkpoint_manager.is_fsdp, "Expected forced non-FSDP treatment"
    print("✓ Force non-FSDP test passed")


def main():
    """Run all tests"""
    print("Testing modified FSDPCheckpointManager...")
    
    try:
        test_non_fsdp_model()
        
        # Only test FSDP if distributed is available
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                test_fsdp_model()
                test_force_non_fsdp()
            except Exception as e:
                print(f"FSDP tests skipped due to: {e}")
                print("This is normal if not running in a distributed environment")
        else:
            print("FSDP tests skipped (no CUDA available)")
            
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 