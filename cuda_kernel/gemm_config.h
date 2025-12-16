#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <cute/tensor.hpp>
#include <cutlass/half.h>

// Refers to https://github.com/reed-lau/cute-gemm/blob/main/gemm-multi-stage.cu
namespace gemm_config {
    using namespace cute;
    template <typename mma_op_, 
              typename ElementA_, typename ElementB_, typename ElementC_,
              int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32,
              int kStage_ = 5, int kSmemLayoutCBatch_ = 2
              >
    struct GemmConfig {
        // tile configuration
        static constexpr int kTileM = kTileM_;
        static constexpr int kTileN = kTileN_;
        static constexpr int kTileK = kTileK_;
        static constexpr int kStage = kStage_;
        static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;

        static constexpr int kShmLoadSwizzleM = 3;
        static constexpr int kShmLoadSwizzleS = 3;
        static constexpr int kShmLoadSwizzleB = 3;

        using SmemLayoutAtom = decltype(composition(
            Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
            make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                        make_stride(Int<kTileK>{}, Int<1>{}))));
        using SmemLayoutA = decltype(
            tile_to_shape(SmemLayoutAtom{},
                            make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
        using SmemLayoutB = decltype(
            tile_to_shape(SmemLayoutAtom{},
                            make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

        using ElementA = ElementA_;
        using ElementB = ElementB_;
        using ElementC = ElementC_;

        using mma_op = mma_op_;

        using mma_traits = MMA_Traits<mma_op>;
        using mma_atom = MMA_Atom<mma_traits>;

        static constexpr int kMmaEURepeatM = 2;
        static constexpr int kMmaEURepeatN = 2;
        static constexpr int kMmaEURepeatK = 1;

        using mma_atom_shape = typename mma_traits::Shape_MNK;
        static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
        static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
        static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

        using MMA_EU_RepeatT = decltype(make_layout(make_shape(
            Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
        using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

        using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

        using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
        using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
        using g2s_copy_atom = Copy_Atom<g2s_copy_traits, ElementA>;

        using G2SCopyA =
            decltype(make_tiled_copy(g2s_copy_atom{},
                                    make_layout(make_shape(Int<32>{}, Int<4>{}),
                                                make_stride(Int<4>{}, Int<1>{})),
                                    make_layout(make_shape(Int<1>{}, Int<8>{}))));
        using G2SCopyB = G2SCopyA;

        // shared memory to register copy
        using s2r_copy_op = SM75_U32x4_LDSM_N;
        using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
        using s2r_copy_atom = Copy_Atom<s2r_copy_traits, ElementA>;

        using S2RCopyAtomA = s2r_copy_atom;
        using S2RCopyAtomB = s2r_copy_atom;

        // epilogue: register to global via shared memory
        using SmemLayoutAtomC = decltype(composition(
            Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                                            make_stride(Int<kMmaPN>{}, Int<1>{}))));
        using SmemLayoutC = decltype(tile_to_shape(
            SmemLayoutAtomC{},
            make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

        static_assert(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) >=
                            size(SmemLayoutC{}),
                        "C shared memory request is large than A's one pipe");

        using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, ElementC>;

        using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, ElementC>;
        using S2GCopyC =
            decltype(make_tiled_copy(S2GCopyAtomC{},
                                    make_layout(make_shape(Int<32>{}, Int<4>{}),
                                                make_stride(Int<4>{}, Int<1>{})),
                                    make_layout(make_shape(Int<1>{}, Int<8>{}))));

        static constexpr int kThreadNum = size(MMA{});
        static constexpr int shm_size_AB =
            cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
        static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});

        static constexpr int kShmSize =
            cute::max(shm_size_AB, shm_size_C) * sizeof(ElementC);
    };
};