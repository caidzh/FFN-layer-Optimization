"""
Environment verification script.
Checks if all dependencies and hardware are properly set up.
"""

import sys

def check_python():
    """Check Python version."""
    print("\n" + "="*70)
    print("Checking Python...")
    print("="*70)
    version = sys.version_info
    print(f"  Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("  ❌ Python 3.7+ required")
        return False
    print("  ✓ Python version OK")
    return True


def check_pytorch():
    """Check PyTorch installation."""
    print("\n" + "="*70)
    print("Checking PyTorch...")
    print("="*70)
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
            
            # Check compute capability
            major, minor = torch.cuda.get_device_capability(0)
            print(f"  Compute capability: {major}.{minor}")
            
            if major < 7:
                print("  ⚠ Warning: Compute capability < 7.0, some optimizations may not work")
            
            print("  ✓ PyTorch with CUDA OK")
            return True
        else:
            print("  ⚠ Warning: CUDA not available, will run on CPU")
            print("  ✓ PyTorch OK (CPU only)")
            return True
            
    except ImportError:
        print("  ❌ PyTorch not installed")
        print("  Install with: pip install torch")
        return False


def check_numpy():
    """Check NumPy installation."""
    print("\n" + "="*70)
    print("Checking NumPy...")
    print("="*70)
    
    try:
        import numpy as np
        print(f"  NumPy version: {np.__version__}")
        print("  ✓ NumPy OK")
        return True
    except ImportError:
        print("  ❌ NumPy not installed")
        print("  Install with: pip install numpy")
        return False


def check_tabulate():
    """Check tabulate installation."""
    print("\n" + "="*70)
    print("Checking tabulate...")
    print("="*70)
    
    try:
        import tabulate
        print(f"  tabulate version: {tabulate.__version__}")
        print("  ✓ tabulate OK")
        return True
    except ImportError:
        print("  ❌ tabulate not installed")
        print("  Install with: pip install tabulate")
        return False


def check_cuda_toolkit():
    """Check CUDA toolkit installation."""
    print("\n" + "="*70)
    print("Checking CUDA Toolkit...")
    print("="*70)
    
    import subprocess
    
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse version from output
            output = result.stdout
            print(f"  NVCC output:\n{output}")
            print("  ✓ CUDA Toolkit installed")
            return True
        else:
            print("  ❌ nvcc command failed")
            return False
    except FileNotFoundError:
        print("  ❌ nvcc not found in PATH")
        print("  CUDA kernel compilation will fail")
        print("  Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        return False
    except Exception as e:
        print(f"  ❌ Error checking CUDA Toolkit: {e}")
        return False


def check_project_structure():
    """Check if project files exist."""
    print("\n" + "="*70)
    print("Checking Project Structure...")
    print("="*70)
    
    import os
    
    required_files = [
        'ffn.py',
        'requirements.txt',
        'README.md',
        'REPORT.md',
        'models/__init__.py',
        'models/ffn_baseline.py',
        'cuda_kernel/__init__.py',
        'cuda_kernel/ffn_cuda.cu',
        'cuda_kernel/ffn_cuda.cpp',
        'cuda_kernel/setup.py',
        'tests/__init__.py',
        'tests/test_correctness.py',
        'tests/test_benchmark.py',
    ]
    
    all_exist = True
    for file in required_files:
        exists = os.path.exists(file)
        status = "✓" if exists else "❌"
        print(f"  {status} {file}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n  ✓ All required files present")
    else:
        print("\n  ❌ Some files missing")
    
    return all_exist


def test_baseline():
    """Test if baseline model works."""
    print("\n" + "="*70)
    print("Testing Baseline Model...")
    print("="*70)
    
    try:
        import torch
        from models.ffn_baseline import GEGLU_FFN
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Testing on: {device}")
        
        # Create model
        ffn = GEGLU_FFN(4096, 12288).to(device)
        print("  ✓ Model created")
        
        # Test forward pass
        x = torch.randn(4, 4096, device=device)
        y = ffn(x)
        print(f"  ✓ Forward pass OK: {x.shape} -> {y.shape}")
        
        # Check output shape
        assert y.shape == (4, 4096), f"Expected shape (4, 4096), got {y.shape}"
        print("  ✓ Output shape correct")
        
        # Check for NaN/Inf
        assert not torch.isnan(y).any(), "Output contains NaN"
        assert not torch.isinf(y).any(), "Output contains Inf"
        print("  ✓ Output numerically stable")
        
        print("\n  ✓ Baseline model works correctly")
        return True
        
    except Exception as e:
        print(f"\n  ❌ Baseline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("\n" + "="*70)
    print(" FFN CUDA OPTIMIZATION - ENVIRONMENT CHECK")
    print("="*70)
    
    results = []
    
    # Run all checks
    results.append(("Python", check_python()))
    results.append(("PyTorch", check_pytorch()))
    results.append(("NumPy", check_numpy()))
    results.append(("tabulate", check_tabulate()))
    results.append(("CUDA Toolkit", check_cuda_toolkit()))
    results.append(("Project Structure", check_project_structure()))
    results.append(("Baseline Model", test_baseline()))
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s} ... {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_checks = len(results)
    
    print(f"\n  Total: {total_passed}/{total_checks} checks passed")
    
    if total_passed == total_checks:
        print("\n  🎉 Environment is ready!")
        print("\n  Next steps:")
        print("    1. Run baseline: python ffn.py")
        print("    2. Implement CUDA kernel: edit cuda_kernel/ffn_cuda.cu")
        print("    3. Compile: cd cuda_kernel && python setup.py build_ext --inplace")
        print("    4. Test: python tests/test_correctness.py")
        print("    5. Benchmark: python tests/test_benchmark.py")
        return 0
    else:
        print("\n  ⚠ Some checks failed. Please fix the issues above.")
        
        # Provide specific guidance
        if not results[1][1]:  # PyTorch
            print("\n  To install PyTorch with CUDA:")
            print("    pip install torch --index-url https://download.pytorch.org/whl/cu118")
        
        if not results[4][1]:  # CUDA Toolkit
            print("\n  To install CUDA Toolkit:")
            print("    https://developer.nvidia.com/cuda-downloads")
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
