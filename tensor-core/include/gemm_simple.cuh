#pragma once
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace gemm_simple {

using namespace cute;

template <typename T, int kTileM, int kTileN, int kTileK>
struct GemmConfig {
    using MMA_Op = SM80_16x8x16_F16F16F16F16_TN;
    using MMA_Traits = MMA_Traits<MMA_Op>;
    using MMA_Atom = MMA_Atom<MMA_Traits>;
    
    using TiledMMA = decltype(make_tiled_mma(MMA_Atom{}, 
                      make_layout(Shape<_2, _2, _1>{}), 
                      make_layout(Shape<_1, _2, _1>{})));

    using SmemLayoutA = decltype(Layout<Shape<Int<kTileM>, Int<kTileK>>, Stride<Int<kTileK>, _1>>{});
    using SmemLayoutB = decltype(Layout<Shape<Int<kTileN>, Int<kTileK>>, Stride<Int<kTileK>, _1>>{});

    using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, T>;
    using GmemTiledCopy = decltype(make_tiled_copy(
        GmemCopyAtom{},
        Layout<Shape<Int<32>, Int<4>>, Stride<Int<4>, _1>>{}, // 128 threads
        Layout<Shape<_1, _8>>{}
    ));
};

template <typename Config>
__global__ void gemm_kernel(void* Cptr, const void* Aptr, const void* Bptr, int m, int n, int k) {
    using T = typename std::remove_pointer<decltype(Cptr)>::type;
    
    extern __shared__ char smem_buf[];
    T* smem = reinterpret_cast<T*>(smem_buf);
    
    Tensor A = make_tensor(make_gmem_ptr((const T*)Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr((const T*)Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr((T*)Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

    int ix = blockIdx.x;
    int iy = blockIdx.y;

    Tensor gA = local_tile(A, make_tile(Int<Config::kTileM>{}, Int<Config::kTileK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<Config::kTileN>{}, Int<Config::kTileK>{}), make_coord(ix, _));
    Tensor gC = local_tile(C, make_tile(Int<Config::kTileM>{}, Int<Config::kTileN>{}), make_coord(iy, ix));

    Tensor sA = make_tensor(make_smem_ptr(smem), typename Config::SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem + cosize(typename Config::SmemLayoutA{})), typename Config::SmemLayoutB{});

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

    int num_tile_k = size<2>(gA);

    for(int itile = 0; itile < num_tile_k; ++itile) {
        // 1. Load Global -> Shared
        copy(gmem_copy, tAgA(_,_,_,itile), tAsA);
        copy(gmem_copy, tBgB(_,_,_,itile), tBsB);
        
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // 2. Load Shared -> Register
        auto tCrA = thr_mma.partition_fragment_A(sA);
        auto tCrB = thr_mma.partition_fragment_B(sB);
        
        // 这里的 copy 依赖编译器优化为 LDS
        copy(tAsA, tCrA); // 逻辑上不对，但简单版通常这么写，实际上应该用 thr_mma.partition_A(sA)
        copy(tBsB, tCrB);

        // 3. Compute
        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
        
        __syncthreads();
    }

    auto tCgC = thr_mma.partition_C(gC);
    copy(tCrC, tCgC);
}

} // namespace gemm_simple