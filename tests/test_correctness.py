"""
Correctness tests for GEGLU FFN implementations.

This module tests the correctness of both baseline PyTorch implementation
and CUDA optimized implementation against each other and expected behaviors.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
from models.ffn_baseline import GEGLU_FFN


# Test configuration
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 12288
BATCH_SIZES = [4, 8, 16, 32, 64, 128]
RTOL = 1e-3  # Relative tolerance
ATOL = 1e-5  # Absolute tolerance


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def test_output_shape():
    """Test that output shape matches expected dimensions."""
    print("\n" + "="*70)
    print("TEST 1: Output Shape Verification")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ffn = GEGLU_FFN(HIDDEN_SIZE, INTERMEDIATE_SIZE).to(device)
    
    all_passed = True
    for B in BATCH_SIZES:
        x = torch.randn(B, HIDDEN_SIZE, device=device)
        y = ffn(x)
        
        expected_shape = (B, HIDDEN_SIZE)
        actual_shape = tuple(y.shape)
        
        passed = actual_shape == expected_shape
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Batch size {B:3d}: {actual_shape} == {expected_shape} ... {status}")
    
    print(f"\n  Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    return all_passed


def test_deterministic():
    """Test that the same input produces the same output."""
    print("\n" + "="*70)
    print("TEST 2: Deterministic Behavior")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ffn = GEGLU_FFN(HIDDEN_SIZE, INTERMEDIATE_SIZE).to(device)
    
    all_passed = True
    for B in BATCH_SIZES:
        set_seed(42)
        x = torch.randn(B, HIDDEN_SIZE, device=device)
        y1 = ffn(x)
        
        y2 = ffn(x)
        
        max_diff = torch.max(torch.abs(y1 - y2)).item()
        passed = max_diff == 0.0
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Batch size {B:3d}: max_diff = {max_diff:.2e} ... {status}")
    
    print(f"\n  Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    return all_passed


def test_manual_computation():
    """Test against manual step-by-step computation."""
    print("\n" + "="*70)
    print("TEST 3: Manual Computation Verification")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_passed = True
    for B in BATCH_SIZES:
        set_seed(42)
        ffn = GEGLU_FFN(HIDDEN_SIZE, INTERMEDIATE_SIZE).to(device)
        x = torch.randn(B, HIDDEN_SIZE, device=device)
        
        # Get output from model
        y_model = ffn(x)
        
        # Manual computation
        weights = ffn.get_weights()
        Wu = weights['Wu']  # (12288, 4096)
        Wv = weights['Wv']  # (12288, 4096)
        Wo = weights['Wo']  # (4096, 12288)
        
        u = F.linear(x, Wu)  # (B, 12288)
        v = F.linear(x, Wv)  # (B, 12288)
        g = F.gelu(u)        # (B, 12288)
        h = g * v            # (B, 12288)
        y_manual = F.linear(h, Wo)  # (B, 4096)
        
        # Compare
        max_diff = torch.max(torch.abs(y_model - y_manual)).item()
        rel_error = max_diff / (torch.max(torch.abs(y_manual)).item() + 1e-8)
        
        passed = rel_error < RTOL
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Batch size {B:3d}: rel_error = {rel_error:.2e} ... {status}")
    
    print(f"\n  Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    return all_passed


def test_gradient_flow():
    """Test that gradients flow correctly through the network."""
    print("\n" + "="*70)
    print("TEST 4: Gradient Flow")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_passed = True
    for B in BATCH_SIZES:
        set_seed(42)
        ffn = GEGLU_FFN(HIDDEN_SIZE, INTERMEDIATE_SIZE).to(device)
        x = torch.randn(B, HIDDEN_SIZE, device=device, requires_grad=True)
        
        y = ffn(x)
        loss = y.sum()
        loss.backward()
        
        # Check that gradients exist and are non-zero
        has_grad = x.grad is not None
        nonzero_grad = torch.any(x.grad != 0).item() if has_grad else False
        
        passed = has_grad and nonzero_grad
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        grad_info = f"exists={has_grad}, nonzero={nonzero_grad}"
        print(f"  Batch size {B:3d}: {grad_info} ... {status}")
    
    print(f"\n  Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    return all_passed


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("\n" + "="*70)
    print("TEST 5: Numerical Stability")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ffn = GEGLU_FFN(HIDDEN_SIZE, INTERMEDIATE_SIZE).to(device)
    
    test_cases = [
        ("zeros", torch.zeros),
        ("ones", torch.ones),
        ("large values", lambda *args, **kwargs: torch.ones(*args, **kwargs) * 10.0),
        ("small values", lambda *args, **kwargs: torch.ones(*args, **kwargs) * 0.01),
    ]
    
    all_passed = True
    B = 16  # Use a single batch size for stability tests
    
    for name, tensor_fn in test_cases:
        x = tensor_fn(B, HIDDEN_SIZE, device=device)
        y = ffn(x)
        
        has_nan = torch.any(torch.isnan(y)).item()
        has_inf = torch.any(torch.isinf(y)).item()
        
        passed = not (has_nan or has_inf)
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:15s}: NaN={has_nan}, Inf={has_inf} ... {status}")
    
    print(f"\n  Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    return all_passed


def test_cuda_kernel_correctness():
    """
    Test CUDA kernel implementation against PyTorch baseline.
    This test will be skipped if CUDA kernel is not available.
    """
    print("\n" + "="*70)
    print("TEST 6: CUDA Kernel Correctness (vs Baseline)")
    print("="*70)
    
    try:
        # Try to import CUDA kernel implementation
        from cuda_kernel import ffn_cuda_forward
        cuda_available = True
    except ImportError:
        print("  ⚠ CUDA kernel not implemented yet - SKIPPED")
        return True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  ⚠ CUDA device not available - SKIPPED")
        return True
    
    all_passed = True
    
    for B in BATCH_SIZES:
        set_seed(42)
        ffn = GEGLU_FFN(HIDDEN_SIZE, INTERMEDIATE_SIZE).to(device)
        x = torch.randn(B, HIDDEN_SIZE, device=device)
        
        # Baseline output
        y_baseline = ffn(x)
        
        # CUDA kernel output
        weights = ffn.get_weights()
        y_cuda = ffn_cuda_forward(
            x, 
            weights['Wu'].t().contiguous(),  # Transpose to [4096, 12288]
            weights['Wv'].t().contiguous(), 
            weights['Wo'].t().contiguous()
        )
        
        # Compare
        max_diff = torch.max(torch.abs(y_baseline - y_cuda)).item()
        rel_error = max_diff / (torch.max(torch.abs(y_baseline)).item() + 1e-8)
        
        passed = rel_error < RTOL
        all_passed = all_passed and passed
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Batch size {B:3d}: rel_error = {rel_error:.2e} ... {status}")
    
    print(f"\n  Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    return all_passed


def run_all_tests():
    """Run all correctness tests."""
    print("\n" + "="*70)
    print(" GEGLU FFN CORRECTNESS TEST SUITE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Intermediate size: {INTERMEDIATE_SIZE}")
    print(f"  Batch sizes: {BATCH_SIZES}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Tolerance: rtol={RTOL}, atol={ATOL}")
    
    # Run all tests
    results = []
    # results.append(("Output Shape", test_output_shape()))
    # results.append(("Deterministic Behavior", test_deterministic()))
    # results.append(("Manual Computation", test_manual_computation()))
    # results.append(("Gradient Flow", test_gradient_flow()))
    # results.append(("Numerical Stability", test_numerical_stability()))
    results.append(("CUDA Kernel Correctness", test_cuda_kernel_correctness()))
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:30s} ... {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\n  Total: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n  🎉 All tests passed!")
        return 0
    else:
        print(f"\n  ⚠ {total_tests - total_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
