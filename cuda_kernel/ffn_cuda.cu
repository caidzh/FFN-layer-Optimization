/*
 * CUDA Kernel for GEGLU FFN Optimization
 * 
 * This file contains the CUDA kernel implementation for optimized FFN computation.
 * 
 * Input:
 *   - x: [B, 4096] input tensor
 *   - Wu: [4096, 12288] weight matrix
 *   - Wv: [4096, 12288] weight matrix
 *   - Wo: [12288, 4096] weight matrix
 * 
 * Output:
 *   - y: [B, 4096] output tensor
 * 
 * Computation:
 *   u = x @ Wu          // [B, 12288]
 *   v = x @ Wv          // [B, 12288]
 *   h = GELU(u) * v     // [B, 12288]
 *   y = h @ Wo          // [B, 4096]
 * 
 * Optimization strategies to consider:
 *   1. Fused matrix multiplication and GELU activation
 *   2. Shared memory optimization for matrix multiplication
 *   3. Warp-level primitives for efficient reduction
 *   4. Memory coalescing for global memory access
 *   5. Kernel fusion to reduce memory bandwidth
 *   6. Tensor Core utilization (if available)
 *   7. Optimal block/grid dimensions
 * 
 * Performance target:
 *   - Achieve at least 3x speedup over PyTorch baseline on A5000
 *   - Maintain numerical accuracy (relative error < 1e-3)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <cute/tensor.hpp>
#include <cutlass/half.h>

using half_t = cutlass::half_t;

#define checkCudaErrors(call)                                      \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__,\
                   cudaGetErrorString(err));                       \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

#define dimX 4096
#define dimY 12288
#define TILE_SIZE 16

// TODO: Implement CUDA kernel here
// 
// Example kernel signature:
// __global__ void ffn_forward_kernel(
//     const float* x,      // Input [B, 4096]
//     const float* Wu,     // Weight [4096, 12288]
//     const float* Wv,     // Weight [4096, 12288]
//     const float* Wo,     // Weight [12288, 4096]
//     float* y,            // Output [B, 4096]
//     int B                // Batch size
// )

// GELU activation function
// GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the standard normal distribution
// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float gelu(float x) {
    // TODO: Implement GELU activation
    // This is a placeholder
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ half_t gelu_half(half_t x) {
    float xf = static_cast<float>(x);
    float result = gelu(xf);
    return static_cast<half_t>(result);
}

__global__ void elementwise_gelu(float* U, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * TILE_SIZE;
    if (idx >= N)
        return;
#pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i)
        U[idx + i] = gelu(U[idx + i]);
}

__global__ void elementwise_mul(float* U, const float* V, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * TILE_SIZE;
    if (idx >= N)
        return;
#pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i)
        U[idx + i] = U[idx + i] * V[idx + i];
}

__global__ void elementwise_gelu_mul_fusion(float *U, const float* V, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * TILE_SIZE;
    if (idx >= N)
        return;
#pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) 
        U[idx + i] = gelu(U[idx + i]) * V[idx + i];
}

__global__ void elementwise_gelu_mul_fusion_half(half_t *U, const half_t* V, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * TILE_SIZE;
    if (idx >= N)
        return;
#pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
        float u_val = static_cast<float>(U[idx + i]);
        float v_val = static_cast<float>(V[idx + i]);
        float result = gelu(u_val) * v_val;
        U[idx + i] = static_cast<half_t>(result);
    }
}

// A cublas implementation of the forward pass for the FFN
void ffn_cuda_forward_launcher_cublas(
    const float* x,
    const float* Wu,
    const float* Wv,
    const float* Wo,
    float* y,
    int B,
    cudaStream_t stream
) {
    dim3 block(32, 1, 1);
    dim3 grid((B * dimY + block.x * TILE_SIZE - 1) / (block.x * TILE_SIZE), 1, 1);
    float *u, *v;
    checkCudaErrors(cudaMalloc((void**)&u, B * dimY * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&v, B * dimY * sizeof(float)));
    cublasHandle_t cublas_handle = nullptr;
    cublasCreate(&cublas_handle);
     // dist[27 x N] = M[27 x 17] * moment[17 x N]
    /* cublasGemmEx(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                sample_num, 27, 17,
                &alpha,
                d_moment_matrix, CUDA_R_32F, sample_num,
                d_transform_matrix, CUDA_R_32F, 17,
                &beta,
                d_dist_matrix, CUDA_R_32F, sample_num,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT);*/
    // u[B x dimY] = x[B x dimX] @ Wu[dimX x dimY]
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dimY, B, dimX,
                &alpha,
                Wu, CUDA_R_32F, dimY,
                x, CUDA_R_32F, dimX,
                &beta,
                u, CUDA_R_32F, dimY,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT);
    // v = x @ Wv
    cublasGemmEx(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dimY, B, dimX,
                &alpha,
                Wv, CUDA_R_32F, dimY,
                x, CUDA_R_32F, dimX,
                &beta,
                v, CUDA_R_32F, dimY,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT);
    // u = GeLU(u) * v
    elementwise_gelu_mul_fusion<<<grid, block>>>(u, v, B * dimY);
    checkCudaErrors(cudaDeviceSynchronize());
    // y[B x dimX] = u[B x dimY] @ Wo[dimY x dimX]
    cublasGemmEx(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dimX, B, dimY,
                &alpha,
                Wo, CUDA_R_32F, dimX,
                u, CUDA_R_32F, dimY,
                &beta,
                y, CUDA_R_32F, dimX,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT);
    cublasDestroy(cublas_handle);
    cudaFree(u);
    cudaFree(v);
    checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

// Refers to https://github.com/reed-lau/cute-gemm/blob/main/gemm-simple.cu
// A block calculate a tile of the matrix multiplication
// This is the most naive implementation of GEMM using CUTE
template <int kTileM, int kTileN, int kTileK, 
          typename ElementA, typename ElementB, typename ElementC,
          typename TiledMMA>
__global__ void GEMM_Tensor(
    const ElementA* ptrA,
    const ElementB* ptrB,
    ElementC* ptrC,
    int m,
    int n,
    int k
) {
    using namespace cute;

    Tensor A = make_tensor(make_gmem_ptr(ptrA), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(ptrB), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(ptrC), make_shape(m, n), make_stride(n, Int<1>{}));

    int ix = blockIdx.x;
    int iy = blockIdx.y;

    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tAgA = thr_mma.partition_A(gA);
    auto tBgB = thr_mma.partition_B(gB);
    auto tCgC = thr_mma.partition_C(gC);

    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _));
 
    clear(tCrC);
  
    int num_tile_k = size<2>(gA);
#pragma unroll 8
    for(int itile = 0; itile < num_tile_k; ++itile) {
        cute::copy(tAgA(_, _, _, itile), tArA);
        cute::copy(tBgB(_, _, _, itile), tBrB);

        cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
    }

    cute::copy(tCrC, tCgC); 
}

// handwritten GEMM kernels
// only test for B >= 128
// input: x [B, dimX], Wu[dimY, dimX], Wv[dimY, dimX], Wo[dimX, dimY]
// output: y [B, dimX]
void ffn_cuda_forward_launcher(
    const half_t* x,
    const half_t* Wu,
    const half_t* Wv,
    const half_t* Wo,
    float* y,
    int B,
    cudaStream_t stream
) {
    half_t *u, *v;
    checkCudaErrors(cudaMalloc((void**)&u, B * dimY * sizeof(half_t)));
    checkCudaErrors(cudaMalloc((void**)&v, B * dimY * sizeof(half_t)));

    using namespace cute;
    // Initialise settings for MMA
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using MMA = decltype(make_tiled_mma(mma_atom{},
                    make_layout(Shape<_2, _2, _1>{}),
                    make_layout(Shape<_1, _2, _1>{})));

    constexpr int kTileM = 128; 
    constexpr int kTileN = 128; 
    constexpr int kTileK = 32; 
    dim3 block(size(MMA{}));
    dim3 grid(dimY / kTileN, B / kTileM);

    // u[B x dimY] = x[B x dimX] @ Wu[dimX x dimY]
    GEMM_Tensor<kTileM, kTileN, kTileK, half_t, half_t, half_t, MMA><<<grid, block>>>(x, Wu, u, B, dimY, dimX);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // v = x @ Wv
    GEMM_Tensor<kTileM, kTileN, kTileK, half_t, half_t, half_t, MMA><<<grid, block>>>(x, Wv, v, B, dimY, dimX);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // u = GeLU(u) * v
    dim3 block1(32, 1, 1);
    dim3 grid1((B * dimY + block1.x * TILE_SIZE - 1) / (block1.x * TILE_SIZE), 1, 1);
    elementwise_gelu_mul_fusion_half<<<grid1, block1>>>(u, v, B * dimY);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Initialise settings for MMA
    using mma_f32_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_f32_traits = MMA_Traits<mma_f32_op>;
    using mma_f32_atom = MMA_Atom<mma_f32_traits>;
    using MMA_f32 = decltype(make_tiled_mma(mma_f32_atom{}, 
                    make_layout(Shape<_2, _2, _1>{}), 
                    make_layout(Shape<_1, _2, _1>{})));

    dim3 block2(size(MMA_f32{}));
    dim3 grid2(dimX / kTileN, B / kTileM);

    // y[B x dimX] = u[B x dimY] @ Wo[dimY x dimX]
    GEMM_Tensor<kTileM, kTileN, kTileK, half_t, half_t, float, MMA_f32><<<grid2, block2>>>(u, Wo, y, B, dimX, dimY);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(u);
    cudaFree(v);
    checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

// calculate [u v] = x @ [Wu Wv] and u = GeLU(u) * v
__global__ void ffn_forward_step1(
    const float* x,
    float* u,
    float* Wu,
    float* Wv,
    int B,
    int 
) {

}

// fuse the first 2 GEMM and GeLU operations
// reserved
void ffn_cuda_forward_launcher_reserved(
    const float* x,
    const float* Wu,
    const float* Wv,
    const float* Wo,
    float* y,
    int B,
    cudaStream_t stream
) {

}
