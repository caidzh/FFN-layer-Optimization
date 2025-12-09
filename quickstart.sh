#!/bin/bash
# Quick start script for FFN CUDA optimization project

echo "======================================"
echo "FFN CUDA Optimization - Quick Start"
echo "======================================"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.7+"
    exit 1
fi
echo "✓ Python found: $(python --version)"

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "⚠ CUDA toolkit not found. CUDA kernel compilation will fail."
else
    echo "✓ CUDA found: $(nvcc --version | grep release)"
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Run baseline
echo ""
echo "======================================"
echo "Running baseline test..."
echo "======================================"
python ffn.py

# Run correctness tests
echo ""
echo "======================================"
echo "Running correctness tests..."
echo "======================================"
python tests/test_correctness.py

# Run benchmark
echo ""
echo "======================================"
echo "Running benchmark tests..."
echo "======================================"
python tests/test_benchmark.py

echo ""
echo "======================================"
echo "Quick start completed!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Implement CUDA kernel in cuda_kernel/ffn_cuda.cu"
echo "2. Compile: cd cuda_kernel && python setup.py build_ext --inplace"
echo "3. Test: python tests/test_correctness.py"
echo "4. Benchmark: python tests/test_benchmark.py"
echo "5. Document your work in REPORT.md"
echo ""
