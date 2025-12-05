/*
 * PyTorch CUDA Extension Binding
 * 
 * This file provides the C++ interface between PyTorch and the CUDA kernel.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declaration of CUDA launcher
void ffn_cuda_forward_launcher(
    const float* x,
    const float* Wu,
    const float* Wv,
    const float* Wo,
    float* y,
    int B,
    cudaStream_t stream
);

// PyTorch binding function
torch::Tensor ffn_cuda_forward(
    torch::Tensor x,      // [B, 4096]
    torch::Tensor Wu,     // [4096, 12288]
    torch::Tensor Wv,     // [4096, 12288]
    torch::Tensor Wo      // [12288, 4096]
) {
    // Check inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(Wu.is_cuda(), "Wu must be a CUDA tensor");
    TORCH_CHECK(Wv.is_cuda(), "Wv must be a CUDA tensor");
    TORCH_CHECK(Wo.is_cuda(), "Wo must be a CUDA tensor");
    
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(x.size(1) == 4096, "x must have shape [B, 4096]");
    
    TORCH_CHECK(Wu.dim() == 2, "Wu must be 2D");
    TORCH_CHECK(Wu.size(0) == 4096 && Wu.size(1) == 12288, "Wu must have shape [4096, 12288]");
    
    TORCH_CHECK(Wv.dim() == 2, "Wv must be 2D");
    TORCH_CHECK(Wv.size(0) == 4096 && Wv.size(1) == 12288, "Wv must have shape [4096, 12288]");
    
    TORCH_CHECK(Wo.dim() == 2, "Wo must be 2D");
    TORCH_CHECK(Wo.size(0) == 12288 && Wo.size(1) == 4096, "Wo must have shape [12288, 4096]");
    
    // Get batch size
    int B = x.size(0);
    
    // Allocate output tensor
    auto y = torch::empty({B, 4096}, x.options());
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch CUDA kernel
    ffn_cuda_forward_launcher(
        x.data_ptr<float>(),
        Wu.data_ptr<float>(),
        Wv.data_ptr<float>(),
        Wo.data_ptr<float>(),
        y.data_ptr<float>(),
        B,
        stream
    );
    
    return y;
}

// Binding definitions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ffn_cuda_forward", &ffn_cuda_forward, "FFN CUDA forward pass");
}
