"""
Setup script for building CUDA extension.

Build and install:
    python setup.py install

Build in-place for development:
    python setup.py build_ext --inplace
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA architecture for A5000 (compute capability 8.6)
# You can modify this based on your GPU
cuda_arch = os.environ.get('TORCH_CUDA_ARCH_LIST', '8.6')

setup(
    name='ffn_cuda',
    ext_modules=[
        CUDAExtension(
            name='ffn_cuda_binding',
            sources=[
                'ffn_cuda_binding.cpp',
                'ffn_cuda.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode', f'arch=compute_86,code=sm_86',  # A5000
                    '-gencode', f'arch=compute_80,code=sm_80',  # A100
                    '-gencode', f'arch=compute_75,code=sm_75',  # RTX 20 series
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
