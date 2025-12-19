#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
// 引入 FP8 类型定义
#include <cutlass/float8.h>


namespace gemm_fp8 {

using namespace cute;

template <typename ElementA_, typename ElementB_, typename ElementC_, 
          int KTileM_, int KTileN_, int KTileK_, int KStage_, int NThreads_>
struct GemmConfig {

    using ElementA = ElementA_; // e.g., float_e4m3_t
    using ElementB = ElementB_; // e.g., float_e4m3_t
    using ElementC = ElementC_; // e.g., float

    static constexpr int kTileM = KTileM_; 
    static constexpr int kTileN = KTileN_; 
    static constexpr int kTileK = KTileK_; 
    static constexpr int NThreads = NThreads_;
    
    // FP8 指令要求 K 维度至少为 32
    static_assert(kTileK % 64 == 0, "kTileK must be a multiple of 64 for multiple buffer FP8 MMA");

    static constexpr int PerRowThreads = kTileK / 16; // FP8: 128bit = 16 elements. 
    // 注意：这里假设 NThreads 足以覆盖一行。如果 kTileK=64, PerRowThreads=4. 
    
    static constexpr int PerRowThreadsWirte = kTileN / 4; // C 是 float (4 bytes), 128bit = 4 elements
    static constexpr int kStage = KStage_;

    // 使用你封装的 SM89 FP8 MMA Atom
    using MMA_Op = SM89_16x8x32_F32E4M3E4M3F32_TN;
    using MMA_Traits = MMA_Traits<MMA_Op>;
    using MMA_Atom = MMA_Atom<MMA_Traits>;
    
    // TiledMMA: 
    // Atom 是 16x8x32 (1个Warp). 
    // 使用 2x2x1 的 Atom 布局 -> 覆盖 32x16x32 的区域，使用 4 个 Warps (128 线程)
    using TiledMMA = decltype(make_tiled_mma(MMA_Atom{}, 
                      make_layout(Shape<_2, _2, _1>{}), 
                      Tile<Int<32>,Int<64>,Int<32>>{})); 

    // --- Shared Memory Layout ---
    // FP8 是 1 字节。为了无冲突访问 128-bit (16字节)，我们需要连续 16 个元素。
    // Swizzle<3, 4, 3>: 
    //   B=3 (8行混合), 
    //   M=4 (2^4 = 16 元素连续), 
    //   S=3 (Shift)
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 4, 3>{},
        make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                    make_stride(Int<kTileK>{}, Int<1>{}))));

    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));

    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

    // --- Global Memory Copy (A/B) ---
    // 128-bit copy = 16 elements of FP8
    using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, ElementA>;
    using GmemTiledCopy = decltype(make_tiled_copy(
          GmemCopyAtom{},
          Layout<Shape<Int<NThreads/PerRowThreads>,Int<PerRowThreads>>,Stride<Int<PerRowThreads>,_1>>{},
          Layout<Shape<_1,_16>>{} // Vectorize 16 elements
    ));

    // --- Register Load (LDSM) ---
    // LDSM 加载 128 bits。对于 FP8，这依然有效，只要 Layout 映射正确。
    using S2RCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
    using S2RCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;

    // --- Epilogue (C) ---
    // C 是 float。
    using SmemLayoutAtomC = decltype(composition(
        Swizzle<3, 3, 3>{}, // float 4字节，4元素=16字节。Swizzle<3,3,3> (8元素) 足够
        make_layout(Shape<Int<kTileM>,Int<kTileN>>{},
                    Stride<Int<kTileN>,_1>{})
    ));

    using SmemLayoutC = decltype(tile_to_shape(
            SmemLayoutAtomC{},
            Shape<Int<kTileM>,Int<kTileN>>{}
    ));

    using CopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, ElementC>;
    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, ElementC>; // Reg -> Smem 使用 int (32bit) 或 vector

    // Gmem Store C: float 128bit = 4 elements
    using GmemCopyC = decltype(make_tiled_copy(
        CopyAtomC{},
        Layout<Shape<Int<NThreads/PerRowThreadsWirte>,Int<PerRowThreadsWirte>>,Stride<Int<PerRowThreadsWirte>,_1>>{},
        Layout<Shape<_1,_4>>{} // Vectorize 4 floats
    ));
};

template <typename Config>
__global__ void gemm_kernel(void* Cptr, const void* Aptr, const void* Bptr, int m, int n, int k) {
    using TA = typename Config::ElementA;
    using TB = typename Config::ElementB;
    using TC = typename Config::ElementC;

    extern __shared__ char smem_buf[];
    
    // A 和 B 使用 FP8 指针
    TA* smem_A = reinterpret_cast<TA*>(smem_buf);
    // B 紧接在 A 之后。注意对齐，虽然 FP8 是 1 字节，但通常 Shared Memory 分配是对齐的。
    // Config::SmemLayoutA::size() 返回的是元素个数。
    
    Tensor A = make_tensor(make_gmem_ptr(reinterpret_cast<const TA*>(Aptr)), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(reinterpret_cast<const TB*>(Bptr)), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(reinterpret_cast<TC*>(Cptr)), make_shape(m, n), make_stride(n, Int<1>{}));

    int ix = blockIdx.x;
    int iy = blockIdx.y;

    Tensor gA = local_tile(A, make_tile(Int<Config::kTileM>{}, Int<Config::kTileK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<Config::kTileN>{}, Int<Config::kTileK>{}), make_coord(ix, _));
    Tensor gC = local_tile(C, make_tile(Int<Config::kTileM>{}, Int<Config::kTileN>{}), make_coord(iy, ix));

    Tensor sA = make_tensor(make_smem_ptr(smem_A), typename Config::SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(sA.data() + sA.size()), typename Config::SmemLayoutB{});

    typename Config::TiledMMA tiled_mma;
    typename Config::GmemTiledCopy gmem_copy;
    
    auto s2r_copy_a = make_tiled_copy_A(typename Config::S2RCopyAtomA{}, tiled_mma);
    auto s2r_copy_b = make_tiled_copy_B(typename Config::S2RCopyAtomB{}, tiled_mma);

    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto thr_copy = gmem_copy.get_slice(threadIdx.x);
    auto thr_s2r_a = s2r_copy_a.get_slice(threadIdx.x);
    auto thr_s2r_b = s2r_copy_b.get_slice(threadIdx.x);

    auto tAgA = thr_copy.partition_S(gA);
    auto tBgB = thr_copy.partition_S(gB);
    auto tAsA = thr_copy.partition_D(sA);
    auto tBsB = thr_copy.partition_D(sB);

    auto tXsA = thr_s2r_a.partition_S(sA);
    auto tXsB = thr_s2r_b.partition_S(sB);

    // 寄存器 Fragment
    auto tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
    auto tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _)); // Accumulator 是 float
    
    auto tCrA_view = thr_s2r_a.retile_D(tCrA);
    auto tCrB_view = thr_s2r_b.retile_D(tCrB);

    clear(tCrC);

    int num_tile_k = size<2>(gA);
    int k_stage = Config::kStage;
    
    // Prologue
    for(int i = 0; i < k_stage - 1; ++i) {
        copy(gmem_copy, tAgA(_,_,_,i), tAsA(_,_,_,i));
        copy(gmem_copy, tBgB(_,_,_,i), tBsB(_,_,_,i));
        cp_async_fence();
    }

    int smem_write_stage = k_stage - 1;
    int smem_read_stage = 0;

    int nk = size<2>(tCrA); 
    cp_async_wait<Config::kStage - 2>();
    __syncthreads();

    copy(s2r_copy_a, tXsA(_,_,0,smem_read_stage), tCrA_view(_,_,0));
    copy(s2r_copy_b, tXsB(_,_,0,smem_read_stage), tCrB_view(_,_,0));

    // Main Loop
    for(int itile = 0; itile < num_tile_k; ++itile) {
        #pragma unroll
        for(int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;
            
            if (ik == nk - 1) {
                cp_async_wait<Config::kStage - 2>();
                __syncthreads();
                smem_read_stage = (smem_read_stage + 1) % k_stage;
            }

            copy(s2r_copy_a, tXsA(_,_,ik_next,smem_read_stage), tCrA_view(_,_,ik_next));
            copy(s2r_copy_b, tXsB(_,_,ik_next,smem_read_stage), tCrB_view(_,_,ik_next));

            if (ik == 0 && itile < num_tile_k - (k_stage - 1)) {
                copy(gmem_copy, tAgA(_,_,_,itile + k_stage - 1), tAsA(_,_,_,smem_write_stage));
                copy(gmem_copy, tBgB(_,_,_,itile + k_stage - 1), tBsB(_,_,_,smem_write_stage));
                cp_async_fence();
                smem_write_stage = (smem_write_stage + 1) % k_stage;
            }

            gemm(tiled_mma, tCrC, tCrA(_,_,ik), tCrB(_,_,ik), tCrC);
        }
    }

    __syncthreads();

    // --- Epilogue ---
    // 复用 Shared Memory 存储 C (float)
    TC* smem_C = reinterpret_cast<TC*>(smem_buf);

    Tensor sC = make_tensor(make_smem_ptr(smem_C), typename Config::SmemLayoutC{});
    auto r2s_tiled_copy_c = make_tiled_copy_C(typename Config::R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(threadIdx.x);

    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrC);
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);

    copy(r2s_tiled_copy_c, tCrC_r2s, tCsC_r2s);

    __syncthreads();

    typename Config::GmemCopyC gmemcopyC;
    auto thr_copy_c = gmemcopyC.get_slice(threadIdx.x);

    auto tCsC_gmem = thr_copy_c.partition_S(sC);
    auto tCgC_gmem = thr_copy_c.partition_D(gC);

    copy(gmemcopyC, tCsC_gmem, tCgC_gmem);
}

} // namespace gemm_fp8