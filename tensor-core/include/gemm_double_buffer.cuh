#pragma once
#include <cute/tensor.hpp>

namespace gemm_double_buffer {

using namespace cute;

template <typename T, int KTileM_, int KTileN_, int KTileK_,int NThreads_>
struct GemmConfig {
    static constexpr int kStage = 2; // Double Buffer
    static constexpr int kTileM = KTileM_; 
    static constexpr int kTileN = KTileN_; 
    static constexpr int kTileK = KTileK_; 
    static constexpr int NThreads = NThreads_;
    static_assert(kTileK%8==0);
    static constexpr int PerRowThreads = kTileK/8; // 4 * 8 = 32

    using ComputeType = T;
    
    using MMA_Op = SM80_16x8x16_F16F16F16F16_TN;
    using MMA_Traits = MMA_Traits<MMA_Op>;
    using MMA_Atom = MMA_Atom<MMA_Traits>;
    
    using TiledMMA = decltype(make_tiled_mma(MMA_Atom{}, 
                      make_layout(Shape<_2, _2, _1>{}), 
                      Tile<Int<32>,Int<32>,Int<16>>{}));

    using SmemLayoutA = decltype(Layout<
        Shape <Int<kTileM>, Int<kTileK>, Int<kStage>>,
        Stride<Int<kTileK>, _1,          Int<kTileM * kTileK>>>{});

    using SmemLayoutB = decltype(Layout<
        Shape <Int<kTileN>, Int<kTileK>, Int<kStage>>,
        Stride<Int<kTileK>, _1,          Int<kTileN * kTileK>>>{});

    using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, T>;
    using GmemTiledCopy = decltype(make_tiled_copy(
          GmemCopyAtom{},
          Layout<Shape<Int<NThreads/PerRowThreads>,Int<PerRowThreads>>,Stride<Int<PerRowThreads>,_1>>{},
          Layout<Shape<_1,_8>>{}
    ));
};

template <typename Config>
__global__ void gemm_kernel(void* Cptr, const void* Aptr, const void* Bptr, int m, int n, int k) {
    using T = typename Config::ComputeType;
    extern __shared__ char smem_buf[];
    T* smem = reinterpret_cast<T*>(smem_buf);

    static_assert(sizeof(T) * Config::kTileK <= 128 ); //确保一个cache line可以放下

    Tensor A = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(Aptr)), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(Bptr)), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(Cptr)), make_shape(m, n), make_stride(n, Int<1>{}));

    int ix = blockIdx.x;
    int iy = blockIdx.y;

    Tensor gA = local_tile(A, make_tile(Int<Config::kTileM>{}, Int<Config::kTileK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<Config::kTileN>{}, Int<Config::kTileK>{}), make_coord(ix, _));
    Tensor gC = local_tile(C, make_tile(Int<Config::kTileM>{}, Int<Config::kTileN>{}), make_coord(iy, ix));

    Tensor sA = make_tensor(make_smem_ptr(smem), typename Config::SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(sA.data() + sA.size()), typename Config::SmemLayoutB{});

    typename Config::TiledMMA tiled_mma;
    typename Config::GmemTiledCopy gmem_copy;
    
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto thr_copy = gmem_copy.get_slice(threadIdx.x);

    auto tAgA = thr_copy.partition_S(gA);
    auto tBgB = thr_copy.partition_S(gB);
    auto tAsA = thr_copy.partition_D(sA);
    auto tBsB = thr_copy.partition_D(sB);

    auto tCrC = thr_mma.partition_fragment_C(gC(_, _));
    clear(tCrC);

    // Prologue: Load first tile
    copy(gmem_copy, tAgA(_,_,_,0), tAsA(_,_,_,0));
    copy(gmem_copy, tBgB(_,_,_,0), tBsB(_,_,_,0));
    cp_async_fence();

    int num_tile_k = size<2>(gA);
    int write_stage = 1;
    int read_stage = 0;

    for(int itile = 0; itile < num_tile_k - 1; ++itile) {
        // 1. Issue Load for Next Tile
        copy(gmem_copy, tAgA(_,_,_,itile+1), tAsA(_,_,_,write_stage));
        copy(gmem_copy, tBgB(_,_,_,itile+1), tBsB(_,_,_,write_stage));
        cp_async_fence();

        // 2. Wait for Current Tile
        cp_async_wait<1>();
        __syncthreads();

        // 3. Compute Current Tile
        auto tCrA = thr_mma.partition_fragment_A(sA(_,_,read_stage));
        auto tCrB = thr_mma.partition_fragment_B(sB(_,_,read_stage));
        
        // 这里的 copy 仍然是普通的 load
        auto tAsA_mma = thr_mma.partition_A(sA);
        auto tBsB_mma = thr_mma.partition_B(sB);
        copy(tAsA_mma(_,_,_,read_stage), tCrA);
        copy(tBsB_mma(_,_,_,read_stage), tCrB);

        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
        
        // 4. Sync to prevent overwriting
        __syncthreads();

        write_stage ^= 1;
        read_stage ^= 1;
    }

    // Epilogue: Compute last tile
    cp_async_wait<0>();
    __syncthreads();
    {
        auto tCrA = thr_mma.partition_fragment_A(sA(_,_,read_stage));
        auto tCrB = thr_mma.partition_fragment_B(sB(_,_,read_stage));
        auto tAsA_mma = thr_mma.partition_A(sA);
        auto tBsB_mma = thr_mma.partition_B(sB);
        copy(tAsA_mma(_,_,_,read_stage), tCrA);
        copy(tBsB_mma(_,_,_,read_stage), tCrB);
        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
    }

    auto tCgC = thr_mma.partition_C(gC);
    copy(tCrC, tCgC);
}

} // namespace gemm_double_buffer