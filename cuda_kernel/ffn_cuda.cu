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
#include "gemm_config.h"

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

// naive handwritten GEMM kernels
// only test for B >= 128
// input: x [B, dimX], Wu[dimY, dimX], Wv[dimY, dimX], Wo[dimX, dimY]
// output: y [B, dimX]
void ffn_cuda_forward_launcher_naive_GEMM(
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
    dim3 grid2((dimX + kTileN - 1) / kTileN, (B + kTileM - 1) / kTileM);

    // y[B x dimX] = u[B x dimY] @ Wo[dimY x dimX]
    GEMM_Tensor<kTileM, kTileN, kTileK, half_t, half_t, float, MMA_f32><<<grid2, block2>>>(u, Wo, y, B, dimX, dimY);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(u);
    cudaFree(v);
    checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

// Refers to https://github.com/reed-lau/cute-gemm/blob/main/gemm-multi-stage.cu
// Efficient implementation of GEMM using Tensor Cores with shared memory optimization and pipelining
template <typename Config>
__global__ void GEMM(
    const void* ptrA,
    const void* ptrB,
    void* ptrC,
    int m,
    int n,
    int k
) {
    using namespace cute;
    using X = Underscore;

    using T_input = typename Config::ElementA;
    using T_output = typename Config::ElementC;
    using SmemLayoutA = typename Config::SmemLayoutA;
    using SmemLayoutB = typename Config::SmemLayoutB;
    using SmemLayoutC = typename Config::SmemLayoutC;
    using TiledMMA = typename Config::MMA;

    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;
    using G2SCopyA = typename Config::G2SCopyA;
    using G2SCopyB = typename Config::G2SCopyB;
    using R2SCopyAtomC = typename Config::R2SCopyAtomC;
    using S2GCopyAtomC = typename Config::S2GCopyAtomC;
    using S2GCopyC = typename Config::S2GCopyC;

    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;
    constexpr int kStage = Config::kStage;

    extern __shared__ T_input shm_data[];

    T_input *Ashm = shm_data;
    T_input *Bshm = shm_data + cute::cosize(SmemLayoutA{});

    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;

    // use Tensor notation to represent device pointer + dimension
    Tensor A = make_tensor(make_gmem_ptr((T_input*)ptrA), make_shape(m, k),
                            make_stride(k, Int<1>{}));  // (M, K)
    Tensor B = make_tensor(make_gmem_ptr((T_input*)ptrB), make_shape(n, k),
                            make_stride(k, Int<1>{}));  // (N, K)
    Tensor C = make_tensor(make_gmem_ptr((T_output*)ptrC), make_shape(m, n),
                            make_stride(n, Int<1>{}));  // (M, N)

    // slice the tensor to small one which is used for current thread block.
    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                            make_coord(iy, _));  // (kTileM, kTileK, k)
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                            make_coord(ix, _));  // (kTileN, kTileK, k)
    Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                            make_coord(iy, ix));  // (kTileM, kTileN)

    // shared memory
    auto sA = make_tensor(make_smem_ptr(Ashm),
                            SmemLayoutA{});  // (kTileM, kTileK, kStage)
    auto sB = make_tensor(make_smem_ptr(Bshm),
                            SmemLayoutB{});  // (kTileN, kTileK, kStage)

    // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition
    // method
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
    auto tCrC = thr_mma.partition_fragment_C(gC);           // (MMA, MMA_M, MMA_N)

    // fill zero for accumulator
    clear(tCrC);

    // gmem -cp.async-> shm -ldmatrix-> reg
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);  // ? (CPY, CPY_M, CPY_K, kStage)
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);  // ? (CPY, CPY_M, CPY_K)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);  // ? (CPY, CPY_M, CPY_K, kStage)
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);  // ? (CPY, CPY_M, CPY_K)

    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
    auto tAsA_copy =
        g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K, k)
    auto tBsB_copy =
        g2s_thr_copy_b.partition_D(sB);  // (CPY, CPY_N, CPY_K, kStage)

    // ring buffer index
    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;

    // submit kStage - 1 tile
    // gmem -> shm
    #pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
                tAsA_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
                tBsB_copy(_, _, _, istage));
        cp_async_fence();

        ++itile_to_read;
        ++ismem_write;
    }

    // wait one submitted gmem->smem done
    // current_group = kStage - 1
    // wait group id <= current_group - (kStage - 2) - 1
    // = current_group - kStage + 1 = 0 done
    cp_async_wait<kStage - 2>();
    __syncthreads();

    int ik = 0;
    // smem -> reg
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

    // loop over k: i. load tile, ii. mma
    int ntile = k / kTileK;
    #pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        int nk = size<2>(tCrA);

    #pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
        int ik_next = (ik + 1) % nk;

        if (ik == nk - 1) {
            cp_async_wait<kStage - 2>();
            __syncthreads();

            ismem_read = (ismem_read + 1) % kStage;
        }

        // shm -> reg s[itile][ik + 1] -> r[ik + 1]
        cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                    tCrA_view(_, _, ik_next));
        cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                    tCrB_view(_, _, ik_next));

        if (ik == 0) {
            if (itile_to_read < ntile) {
            cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                        tAsA_copy(_, _, _, ismem_write));
            cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                        tBsB_copy(_, _, _, ismem_write));

            ++itile_to_read;
            ismem_write = (ismem_write + 1) % kStage;
            }

            cp_async_fence();
        }

        cute::gemm(tiled_mma, tCrC, tCrA(_, _, ik), tCrB(_, _, ik), tCrC);
        }  // for ik
    }    // itile

    // use less shared memory as a scratchpad tile to use large wide instuction
    // Creg -> shm -> reg -> global
    auto sC = make_tensor(sB(_, _, ismem_read).data(), SmemLayoutC{});

    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrC);   // (CPY, CPY_M, CPY_N)
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  // (CPY, _1, _1, pipe)

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);  // (CPY, _1, _1, pipe)
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gC);  // (CPY, CPY_M, CPY_N)

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);  // (CPY_, CPY_MN)
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN)

    int step = size<3>(tCsC_r2s);  // pipe
    #pragma unroll
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
        // reg -> shm
    #pragma unroll
        for (int j = 0; j < step; ++j) {
        // we add a temp tensor to cope with accumulator and output data type
        // difference
        auto t = make_tensor_like<T_output>(tCrC_r2sx(_, i + j));
        cute::copy(tCrC_r2sx(_, i + j), t);

        cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();

    #pragma unroll
        // shm -> global
        for (int j = 0; j < step; ++j) {
        cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }

        __syncthreads();
    }

}

// handwritten GEMM kernels using Tensor Cores with shared memory optimization and pipelining
// only test for B >= 128
// input: x [B, dimX], Wu[dimY, dimX], Wv[dimY, dimX], Wo[dimY, dimX] (Wo is transposed)
// output: y [B, dimX]
void ffn_cuda_forward_launcher_efficient_GEMM(
    const half_t* x,
    const half_t* Wu,
    const half_t* Wv,
    const half_t* Wo,
    float* y,
    int B,
    cudaStream_t stream
) {
    using namespace cute;

    half_t *u, *v;
    checkCudaErrors(cudaMalloc((void**)&u, B * dimY * sizeof(half_t)));
    checkCudaErrors(cudaMalloc((void**)&v, B * dimY * sizeof(half_t)));

    // Initialise settings for MMA
    gemm_config::GemmConfig<
            SM80_16x8x16_F16F16F16F16_TN,
            half_t, half_t, half_t,
            64, 64, 32, 3> config;

    dim3 block = config.kThreadNum;
    dim3 grid((dimY + config.kTileN - 1) / config.kTileN,
                (B + config.kTileM - 1) / config.kTileM);
    int shm_size = config.kShmSize;

    cudaFuncSetAttribute(GEMM<decltype(config)>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    // u = x @ Wu
    GEMM<decltype(config)><<<grid, block, shm_size>>>(x, Wu, u, B, dimY, dimX);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // v = x @ Wv
    GEMM<decltype(config)><<<grid, block, shm_size>>>(x, Wv, v, B, dimY, dimX);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // u = GeLU(u) * v
    dim3 block1(32, 1, 1);
    dim3 grid1((B * dimY + block1.x * TILE_SIZE - 1) / (block1.x * TILE_SIZE), 1, 1);
    elementwise_gelu_mul_fusion_half<<<grid1, block1>>>(u, v, B * dimY);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    /*
    // Initialise settings for MMA
    gemm_config::GemmConfig<
            SM80_16x8x16_F32F16F16F32_TN,
            half_t, half_t, float,
            128, 128, 32, 3> config2;

    dim3 block2 = config2.kThreadNum;
    dim3 grid2((dimX + config2.kTileN - 1) / config2.kTileN,
                (B + config2.kTileM - 1) / config2.kTileM);
    int shm_size2 = config2.kShmSize;

    cudaFuncSetAttribute(GEMM<decltype(config2)>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size2);

    // y = u @ Wo
    GEMM<decltype(config2)><<<grid2, block2, shm_size2>>>(u, Wo, y, B, dimX, dimY);
    */
    cublasHandle_t cublas_handle = nullptr;
    cublasCreate(&cublas_handle);
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dimX, B, dimY,
                &alpha,
                Wo, CUDA_R_16F, dimX,
                u, CUDA_R_16F, dimY,
                &beta,
                y, CUDA_R_32F, dimX,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cublasDestroy(cublas_handle);
    cudaFree(u);
    cudaFree(v);
    checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

// calculate [u v] = x @ [Wu Wv] and u = GeLU(u) * v
template <typename Config>
__global__ void matmul_gelu_fuse_kernel(
    const void* x,
    const void* Wu,
    const void* Wv,
    void* u,
    void* v,
    int m,
    int n,
    int k
) {
    using namespace cute;
    using X = Underscore;

    using T_input = typename Config::ElementA;
    using T_output = typename Config::ElementC;
    using SmemLayoutA = typename Config::SmemLayoutA;
    using SmemLayoutB = typename Config::SmemLayoutB;
    using SmemLayoutC = typename Config::SmemLayoutC;
    using TiledMMA = typename Config::MMA;

    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;
    using G2SCopyA = typename Config::G2SCopyA;
    using G2SCopyB = typename Config::G2SCopyB;
    using R2SCopyAtomC = typename Config::R2SCopyAtomC;
    using S2GCopyAtomC = typename Config::S2GCopyAtomC;
    using S2GCopyC = typename Config::S2GCopyC;

    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;
    constexpr int kStage = Config::kStage;

    extern __shared__ T_input shm_data[];

    T_input *Ashm = shm_data;
    T_input *Bshm1 = shm_data + cute::cosize(SmemLayoutA{});
    T_input *Bshm2 = Bshm1 + cute::cosize(SmemLayoutB{});

    int idx = threadIdx.x;
    int ix = blockIdx.x;
    int iy = blockIdx.y;

    // use Tensor notation to represent device pointer + dimension
    Tensor A = make_tensor(make_gmem_ptr((T_input*)x), make_shape(m, k),
                            make_stride(k, Int<1>{}));  // (M, K)
    Tensor B1 = make_tensor(make_gmem_ptr((T_input*)Wu), make_shape(n, k),
                            make_stride(k, Int<1>{}));  // (N, K)
    Tensor B2 = make_tensor(make_gmem_ptr((T_input*)Wv), make_shape(n, k),
                            make_stride(k, Int<1>{}));  // (N, K)
    Tensor C1 = make_tensor(make_gmem_ptr((T_output*)u), make_shape(m, n),
                            make_stride(n, Int<1>{}));  // (M, N)
    Tensor C2 = make_tensor(make_gmem_ptr((T_output*)v), make_shape(m, n),
                            make_stride(n, Int<1>{}));  // (M, N)

    // slice the tensor to small one which is used for current thread block.
    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}),
                            make_coord(iy, _));  // (kTileM, kTileK, k)
    Tensor gB1 = local_tile(B1, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                            make_coord(ix, _));  // (kTileN, kTileK, k)
    Tensor gB2 = local_tile(B2, make_tile(Int<kTileN>{}, Int<kTileK>{}),
                            make_coord(ix, _));  // (kTileN, kTileK, k)
    Tensor gC1 = local_tile(C1, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                            make_coord(iy, ix));  // (kTileM, kTileN)
    Tensor gC2 = local_tile(C2, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                            make_coord(iy, ix));  // (kTileM, kTileN)

    // shared memory
    auto sA = make_tensor(make_smem_ptr(Ashm),
                            SmemLayoutA{});  // (kTileM, kTileK, kStage)
    auto sB1 = make_tensor(make_smem_ptr(Bshm1),
                            SmemLayoutB{});  // (kTileN, kTileK, kStage)
    auto sB2 = make_tensor(make_smem_ptr(Bshm2),
                            SmemLayoutB{});  // (kTileN, kTileK, kStage)

    // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition
    // method
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    auto tCrB1 = thr_mma.partition_fragment_B(gB1(_, _, 0));  // (MMA, MMA_N, MMA_K)
    auto tCrB2 = thr_mma.partition_fragment_B(gB2(_, _, 0));  // (MMA, MMA_N, MMA_K)
    auto tCrC1 = thr_mma.partition_fragment_C(gC1);           // (MMA, MMA_M, MMA_N)
    auto tCrC2 = thr_mma.partition_fragment_C(gC2);           // (MMA, MMA_M, MMA_N)

    // fill zero for accumulator
    clear(tCrC1);
    clear(tCrC2);

    // gmem -cp.async-> shm -ldmatrix-> reg
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);  // ? (CPY, CPY_M, CPY_K, kStage)
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);  // ? (CPY, CPY_M, CPY_K)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tB1sB1 = s2r_thr_copy_b.partition_S(sB1);  // ? (CPY, CPY_M, CPY_K, kStage)
    auto tB2sB2 = s2r_thr_copy_b.partition_S(sB2);  // ? (CPY, CPY_M, CPY_K, kStage)
    auto tCrB1_view = s2r_thr_copy_b.retile_D(tCrB1);  // ? (CPY, CPY_M, CPY_K)
    auto tCrB2_view = s2r_thr_copy_b.retile_D(tCrB2);  // ? (CPY, CPY_M, CPY_K)

    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
    auto tAsA_copy =
        g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)·

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tB1gB1_copy = g2s_thr_copy_b.partition_S(gB1);  // (CPY, CPY_N, CPY_K, k)
    auto tB2gB2_copy = g2s_thr_copy_b.partition_S(gB2);  // (CPY, CPY_N, CPY_K, k)
    auto tB1sB1_copy =
        g2s_thr_copy_b.partition_D(sB1);  // (CPY, CPY_N, CPY_K, kStage)
    auto tB2sB2_copy =
        g2s_thr_copy_b.partition_D(sB2);  // (CPY, CPY_N, CPY_K, kStage)

    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;

    // submit kStage - 1 tile
    // gmem -> shm
    #pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
                tAsA_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, tB1gB1_copy(_, _, _, istage),
                tB1sB1_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, tB2gB2_copy(_, _, _, istage),
                tB2sB2_copy(_, _, _, istage));
        cp_async_fence();

        ++itile_to_read;
        ++ismem_write;
    }

    // wait one submitted gmem->smem done
    cp_async_wait<kStage - 2>();
    __syncthreads();

    int ik = 0;
    // smem -> reg
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
    cute::copy(s2r_tiled_copy_b, tB1sB1(_, _, ik, ismem_read), tCrB1_view(_, _, ik));
    cute::copy(s2r_tiled_copy_b, tB2sB2(_, _, ik, ismem_read), tCrB2_view(_, _, ik));

    // loop over k: i. load tile, ii. mma
    int ntile = k / kTileK;
    #pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        int nk = size<2>(tCrA);

    #pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
        int ik_next = (ik + 1) % nk;

        if (ik == nk - 1) {
            cp_async_wait<kStage - 2>();
            __syncthreads();

            ismem_read = (ismem_read + 1) % kStage;
        }

        // shm -> reg s[itile][ik + 1] -> r[ik + 1]
        cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                    tCrA_view(_, _, ik_next));
        cute::copy(s2r_tiled_copy_b, tB1sB1(_, _, ik_next, ismem_read),
                    tCrB1_view(_, _, ik_next));
        cute::copy(s2r_tiled_copy_b, tB2sB2(_, _, ik_next, ismem_read),
                    tCrB2_view(_, _, ik_next));

        if (ik == 0) {
            if (itile_to_read < ntile) {
            cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                        tAsA_copy(_, _, _, ismem_write));
            cute::copy(g2s_tiled_copy_b, tB1gB1_copy(_, _, _, itile_to_read),
                        tB1sB1_copy(_, _, _, ismem_write));
            cute::copy(g2s_tiled_copy_b, tB2gB2_copy(_, _, _, itile_to_read),
                        tB2sB2_copy(_, _, _, ismem_write));

            ++itile_to_read;
            ismem_write = (ismem_write + 1) % kStage;
            }

            cp_async_fence();
        }

        cute::gemm(tiled_mma, tCrC1, tCrA(_, _, ik), tCrB1(_, _, ik), tCrC1);
        cute::gemm(tiled_mma, tCrC2, tCrA(_, _, ik), tCrB2(_, _, ik), tCrC2);
        }  // for ik
    }    // itile
    __syncthreads();

#pragma unroll
    for (int i = 0; i < size(tCrC1); ++i) {
        T_input val1 = static_cast<T_input>(tCrC1(i));
        T_input val2 = static_cast<T_input>(tCrC2(i));
        tCrC1(i) = static_cast<T_output>(gelu(val1) * val2);
    }
    __syncthreads();

    // use less shared memory as a scratchpad tile to use large wide instuction
    // Creg -> shm -> reg -> global
    auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrC1);   // (CPY, CPY_M, CPY_N)
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  // (CPY, _1, _1, pipe)

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);  // (CPY, _1, _1, pipe)
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gC1);  // (CPY, CPY_M, CPY_N)

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);  // (CPY_, CPY_MN)
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN)

    int step = size<3>(tCsC_r2s);  // pipe
    #pragma unroll
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
        // reg -> shm
    #pragma unroll
        for (int j = 0; j < step; ++j) {
        // we add a temp tensor to cope with accumulator and output data type
        // difference
        auto t = make_tensor_like<T_output>(tCrC_r2sx(_, i + j));
        cute::copy(tCrC_r2sx(_, i + j), t);

        cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();

    #pragma unroll
        // shm -> global
        for (int j = 0; j < step; ++j) {
        cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }

        __syncthreads();
    }
}

// fuse the first 2 GEMM and GeLU operations
// only test for B >= 128
// input: x [B, dimX], Wu[dimY, dimX], Wv[dimY, dimX], Wo[dimY, dimX] (Wo is transposed)
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
    using namespace cute;

    half_t *u, *v;
    checkCudaErrors(cudaMalloc((void**)&u, B * dimY * sizeof(half_t)));
    checkCudaErrors(cudaMalloc((void**)&v, B * dimY * sizeof(half_t)));

    // Initialise settings for MMA
    gemm_config::GemmConfig<
            SM80_16x8x16_F16F16F16F16_TN,
            half_t, half_t, half_t,
            16, 128, 32, 3> config;

    dim3 block = config.kThreadNum;
    dim3 grid((dimY + config.kTileN - 1) / config.kTileN,
                (B + config.kTileM - 1) / config.kTileM);
    int shm_size = config.kShmSize;

    printf("%d\n", shm_size);

    cudaFuncSetAttribute(GEMM<decltype(config)>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

    // u = x @ Wu
    matmul_gelu_fuse_kernel<decltype(config)><<<grid, block, shm_size>>>(x, Wu, Wv, u, v, B, dimY, dimX);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cublasHandle_t cublas_handle = nullptr;
    cublasCreate(&cublas_handle);
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(cublas_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dimX, B, dimY,
                &alpha,
                Wo, CUDA_R_16F, dimX,
                u, CUDA_R_16F, dimY,
                &beta,
                y, CUDA_R_32F, dimX,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT);
    cublasDestroy(cublas_handle);
    cudaFree(u);
    cudaFree(v);
}
