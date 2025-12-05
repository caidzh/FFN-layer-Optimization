# CUDA Kernel Implementation Guide

This directory contains the CUDA kernel implementation files for FFN.

## File Description

- `ffn_cuda.cu` - CUDA kernel implementation
- `ffn_cuda.cpp` - PyTorch C++ extension interface
- `setup.py` - Build configuration file
- `__init__.py` - Python package initialization

## Building CUDA Extension

### Development Mode (Recommended)
```bash
cd cuda_kernel
python setup.py build_ext --inplace
```

### Installation Mode
```bash
cd cuda_kernel
python setup.py install
```

## Implementation

### 1. CUDA Kernel Implementation (ffn_cuda.cu)

Core functionality to implement:

```cuda
__global__ void ffn_forward_kernel(
    const float* x,      // Input [B, 4096]
    const float* Wu,     // Weight [4096, 12288]
    const float* Wv,     // Weight [4096, 12288]
    const float* Wo,     // Weight [12288, 4096]
    float* y,            // Output [B, 4096]
    int B                // Batch size
)
```

### 3. Performance Testing

After compilation, run:

```bash
# Correctness test
python tests/test_correctness.py

# Performance benchmark
python tests/test_benchmark.py
```

## Expected Performance

Target performance on NVIDIA A5000:

| Batch Size | Baseline (ms) | Target (ms) | Speedup |
|------------|---------------|-------------|---------|
| 4          | ~X.XX         | <X.XX/3     | ≥3x     |
| 8          | ~X.XX         | <X.XX/3     | ≥3x     |
| 16         | ~X.XX         | <X.XX/3     | ≥3x     |
| 32         | ~X.XX         | <X.XX/3     | ≥3x     |
| 64         | ~X.XX         | <X.XX/3     | ≥3x     |
| 128        | ~X.XX         | <X.XX/3     | ≥3x     |