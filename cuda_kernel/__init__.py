"""
CUDA kernel package for optimized FFN implementation.

This package will contain the CUDA kernel implementation and Python bindings.
"""

# Placeholder - will be implemented with actual CUDA kernel
__all__ = []

try:
    from .ffn_cuda_binding import ffn_cuda_forward
    __all__.append('ffn_cuda_forward')
except ImportError:
    # CUDA extension not built yet
    pass
