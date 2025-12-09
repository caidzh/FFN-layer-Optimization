# FFN CUDA Optimization

A CUDA kernel implementation project for optimizing Feed-Forward Network (FFN) computation in LLMs.

## Project Overview

This project implements a CUDA-optimized version of GEGLU FFN, targeting at least 3x speedup on NVIDIA A5000 GPU.

### Specifications
- **Hidden Size**: 4096
- **Intermediate Size**: 12288
- **Batch Sizes**: 4, 8, 16, 32, 64, 128
- **FFN Type**: GEGLU (Gated GLU with GELU activation)

### GEGLU FFN Formula

```
u = Wu @ x
v = Wv @ x
h = GELU(u) ⊙ v
y = Wo @ h
```

Where:
- Wu: [4096, 12288] weight matrix
- Wv: [4096, 12288] weight matrix
- Wo: [12288, 4096] weight matrix
- ⊙ denotes element-wise multiplication

## Project Structure

```
.
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── ffn.py # Baseline PyTorch implementation
├── check_env.py # Check dependencies
├── quickstart.sh
├── models/
│   ├── init.py
│   └── ffn_baseline.py # Refactored baseline module
├── cuda_kernel/
│   ├── init.py
│   ├── README.md #  CUDA kernel documentation
│   ├── ffn_cuda.cu # CUDA kernel implementation (to be implemented)
│   ├── ffn_cuda.cpp # PyTorch binding (to be implemented)
│   └── setup.py # CUDA build configuration (to be implemented)
└── tests/
    ├── init.py
    ├── test_correctness.py # Correctness tests
    └── test_benchmark.py # Performance benchmarks
```

## Installation

```bash
pip install -r requirements.txt
```

### Environment Verification

After installation, run the environment check script:

```bash
python check_env.py
```

This will check:
- Python version
- PyTorch and CUDA
- Required Python packages
- CUDA Toolkit
- Project file structure
- Baseline model functionality

## Running Tests

### 1. Baseline Performance Test
```bash
python ffn.py
```

### 2. Correctness Verification
```bash
python tests/test_correctness.py
```

### 3. Performance Comparison
```bash
python tests/test_benchmark.py
```

## Development Guide

### Baseline Implementation

The baseline uses PyTorch's standard nn.Linear layers, located in `ffn.py` and `ffn_baseline.py`.

### CUDA Kernel Implementation

CUDA kernels need to be implemented in the cuda_kernel directory, including:

1. `ffn_cuda.cu` - CUDA kernel code
2. `ffn_cuda.cpp` - PyTorch C++ extension interface
3. `setup.py` - Build configuration

### Optimization Goals
- Achieve at least 3x speedup across all batch sizes
- Maintain numerical precision (relative error < 1e-3)
- Memory efficient

## Performance Baseline

Running the baseline will output execution time for each batch size. The CUDA-optimized version needs to achieve at least 3x speedup over these baselines.
