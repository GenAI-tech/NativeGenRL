#!/usr/bin/env python3
"""
Test script to check if model parameters are changing across different global steps.
This script loads checkpoints from different global steps and compares their parameters.
"""

import os
import torch
import numpy as np
from collections import OrderedDict
import argparse
from typing import Dict, List, Tuple


def load_model_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load model state dict from checkpoint file."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"Loaded {len(state_dict)} parameters")
    return state_dict


def compute_parameter_stats(state_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute statistics for all parameters in the state dict."""
    stats = {}
    total_params = 0
    total_norm = 0.0
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            param_count = param.numel()
            param_norm = param.norm().item()
            param_mean = param.mean().item()
            param_std = param.std().item()
            
            stats[name] = {
                'count': param_count,
                'norm': param_norm,
                'mean': param_mean,
                'std': param_std,
                'shape': list(param.shape)
            }
            
            total_params += param_count
            total_norm += param_norm ** 2
    
    stats['_TOTAL_'] = {
        'count': total_params,
        'norm': np.sqrt(total_norm),
        'mean': 0.0,  # Not meaningful for total
        'std': 0.0    # Not meaningful for total
    }
    
    return stats


def compare_parameters(state_dict1: Dict[str, torch.Tensor], 
                      state_dict2: Dict[str, torch.Tensor],
                      step1: int, step2: int) -> Dict[str, float]:
    """Compare parameters between two state dicts."""
    print(f"\n=== Comparing Global Step {step1} vs Global Step {step2} ===")
    
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        print("WARNING: Parameter names don't match between checkpoints!")
        common_keys = set(state_dict1.keys()) & set(state_dict2.keys())
        print(f"Using {len(common_keys)} common parameters")
    else:
        common_keys = set(state_dict1.keys())
        print(f"Comparing {len(common_keys)} parameters")
    
    differences = {}
    total_change_norm = 0.0
    total_relative_change = 0.0
    significant_changes = 0
    
    for name in sorted(common_keys):
        param1 = state_dict1[name]
        param2 = state_dict2[name]
        
        if not isinstance(param1, torch.Tensor) or not isinstance(param2, torch.Tensor):
            continue
            
        if param1.shape != param2.shape:
            print(f"WARNING: Shape mismatch for {name}: {param1.shape} vs {param2.shape}")
            continue
        
        # Compute difference
        diff = param2 - param1
        diff_norm = diff.norm().item()
        param1_norm = param1.norm().item()
        
        # Relative change (avoid division by zero)
        relative_change = diff_norm / (param1_norm + 1e-8)
        
        differences[name] = {
            'abs_change': diff_norm,
            'relative_change': relative_change,
            'param1_norm': param1_norm,
            'param2_norm': param2.norm().item()
        }
        
        total_change_norm += diff_norm ** 2
        total_relative_change += relative_change
        
        # Count significant changes (relative change > 1e-6)
        if relative_change > 1e-6:
            significant_changes += 1
    
    total_change_norm = np.sqrt(total_change_norm)
    avg_relative_change = total_relative_change / len(common_keys)
    
    print(f"Total parameter change norm: {total_change_norm:.6e}")
    print(f"Average relative change: {avg_relative_change:.6e}")
    print(f"Parameters with significant changes (>1e-6): {significant_changes}/{len(common_keys)}")
    
    # Show top 10 parameters with largest changes
    sorted_diffs = sorted(differences.items(), key=lambda x: x[1]['relative_change'], reverse=True)
    print(f"\nTop 10 parameters with largest relative changes:")
    for i, (name, diff_info) in enumerate(sorted_diffs[:10]):
        print(f"  {i+1:2d}. {name:50s} | Rel: {diff_info['relative_change']:.6e} | Abs: {diff_info['abs_change']:.6e}")
    
    return {
        'total_change_norm': total_change_norm,
        'avg_relative_change': avg_relative_change,
        'significant_changes': significant_changes,
        'total_parameters': len(common_keys)
    }


def main():
    parser = argparse.ArgumentParser(description='Test checkpoint parameter changes')
    parser.add_argument('--base_path', type=str, 
                       default='NativeGenRl/ckpt/ngrl/20250806_193043',
                       help='Base path to checkpoints')
    parser.add_argument('--steps', type=int, nargs='+', default=[1, 2, 3],
                       help='Global steps to compare')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed parameter statistics')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CHECKPOINT PARAMETER CHANGE ANALYSIS")
    print("=" * 80)
    print(f"Base path: {args.base_path}")
    print(f"Global steps: {args.steps}")
    
    # Load all checkpoints
    checkpoints = {}
    for step in args.steps:
        checkpoint_path = os.path.join(args.base_path, f'global_step_{step}', 'actor', 'model_world_size_4.pt')
        try:
            checkpoints[step] = load_model_checkpoint(checkpoint_path)
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            continue
    
    if len(checkpoints) < 2:
        print("ERROR: Need at least 2 checkpoints to compare!")
        return
    
    available_steps = sorted(checkpoints.keys())
    print(f"Successfully loaded checkpoints for steps: {available_steps}")
    
    # Compute statistics for each checkpoint
    if args.detailed:
        print("\n" + "=" * 80)
        print("PARAMETER STATISTICS FOR EACH CHECKPOINT")
        print("=" * 80)
        
        for step in available_steps:
            print(f"\n--- Global Step {step} ---")
            stats = compute_parameter_stats(checkpoints[step])
            total_stats = stats['_TOTAL_']
            print(f"Total parameters: {total_stats['count']:,}")
            print(f"Total parameter norm: {total_stats['norm']:.6e}")
            
            # Show stats for a few key parameters
            param_names = [name for name in stats.keys() if not name.startswith('_')][:5]
            for name in param_names:
                param_stats = stats[name]
                print(f"  {name:40s} | Shape: {param_stats['shape']} | Norm: {param_stats['norm']:.6e}")
    
    # Compare consecutive checkpoints
    print("\n" + "=" * 80)
    print("PARAMETER CHANGE ANALYSIS")
    print("=" * 80)
    
    comparison_results = []
    for i in range(len(available_steps) - 1):
        step1, step2 = available_steps[i], available_steps[i + 1]
        result = compare_parameters(checkpoints[step1], checkpoints[step2], step1, step2)
        comparison_results.append((step1, step2, result))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if not comparison_results:
        print("No comparisons performed!")
        return
    
    print("Step Comparison | Total Change Norm | Avg Rel Change | Significant Changes")
    print("-" * 75)
    for step1, step2, result in comparison_results:
        print(f"{step1:4d} -> {step2:4d}     | {result['total_change_norm']:15.6e} | "
              f"{result['avg_relative_change']:12.6e} | "
              f"{result['significant_changes']:6d}/{result['total_parameters']:6d}")
    
    # Check if model is actually training
    max_change = max([result['avg_relative_change'] for _, _, result in comparison_results])
    min_change = min([result['avg_relative_change'] for _, _, result in comparison_results])
    
    print(f"\nMax average relative change: {max_change:.6e}")
    print(f"Min average relative change: {min_change:.6e}")
    
    if max_change < 1e-8:
        print("⚠️  WARNING: Very small parameter changes detected!")
        print("   This might indicate:")
        print("   - Model is not training properly")
        print("   - Learning rate is too small")
        print("   - Gradients are not flowing")
    elif max_change > 1e-3:
        print("✅ GOOD: Significant parameter changes detected - model is training!")
    elif max_change > 1e-6:
        print("🔍 MODERATE: Some parameter changes detected - training might be slow")
    else:
        print("❓ UNCLEAR: Small but non-zero changes - check if this is expected")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()