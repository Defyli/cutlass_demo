#pragma once
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace gemm_final_opt {

using namespace cute;

template <typename T, int KTileM_, int KTileN_, int KTileK_, int KStage_, int NThreads_>
struct GemmConfig {

    static constexpr int kTileM = KTileM_; 
    static constexpr int kTileN = KTileN_; 
    static constexpr int kTileK = KTileK_; 
    static constexpr int NThreads = NThreads_;
    static_assert(kTileK%8==0);
    static constexpr int PerRowThreads = kTileK/8; // 4 * 8 = 32
    static constexpr int PerRowThreadsWirte = kTileN/8; 
    static constexpr int kStage = KStage_;

    using ComputeType = T;
    using MMA_Op = SM80_16x8x16_F16F16F16F16_TN;
    using MMA_Traits = MMA_Traits<MMA_Op>;
    using MMA_Atom = MMA_Atom<MMA_Traits>;
    
    using TiledMMA = decltype(make_tiled_mma(MMA_Atom{}, 
                      make_layout(Shape<_2, _2, _1>{}), 
                      Tile<Int<32>,Int<32>,Int<16>>{})); // K-dim 2 for better interleaving

    // Swizzle Layout for Shared Memory
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                    make_stride(Int<kTileK>{}, Int<1>{}))));

    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));

    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

    using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, T>;
    using GmemTiledCopy = decltype(make_tiled_copy(
          GmemCopyAtom{},
          Layout<Shape<Int<NThreads/PerRowThreads>,Int<PerRowThreads>>,Stride<Int<PerRowThreads>,_1>>{},
          Layout<Shape<_1,_8>>{}
    ));

    // Ldmatrix Copy Atoms
    // A: Row-Major Smem -> Row-Major Reg (Normal)
    using S2RCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, T>;
    // B: Row-Major Smem -> Col-Major Reg (Normal, because Smem K is continuous)
    using S2RCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, T>;

    // C copy reg-->smem --> global mem
    using SmemLayoutAtomC = decltype(composition(
        Swizzle<3,3,3>{},
        make_layout(Shape<Int<kTileM>,Int<kTileN>>{},
                    Stride<Int<kTileN>,_1>{})
    ));

    using SmemLayoutC = decltype(tile_to_shape(
            SmemLayoutAtomC{},
            Shape<Int<kTileM>,Int<kTileN>>{}
    ));

    using CopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;

    using GmemCopyC = decltype(make_tiled_copy(
        CopyAtomC{},
        Layout<Shape<Int<NThreads/PerRowThreadsWirte>,Int<PerRowThreadsWirte>>,Stride<Int<PerRowThreadsWirte>,_1> 
        >{},
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
    
    // S2R Tiled Copy
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

    // S2R Views
    auto tXsA = thr_s2r_a.partition_S(sA);
    auto tXsB = thr_s2r_b.partition_S(sB);

    // Registers
    auto tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
    auto tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _));
    
    // Retile for ldmatrix
    auto tCrA_view = thr_s2r_a.retile_D(tCrA);
    auto tCrB_view = thr_s2r_b.retile_D(tCrB);

    clear(tCrC);

    // Prologue
    int num_tile_k = size<2>(gA);
    int k_stage = Config::kStage;
    
    // Prefetch stages
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

    // Prologue: 预加载第一个 K-slice 到寄存器
    copy(s2r_copy_a, tXsA(_,_,0,smem_read_stage), tCrA_view(_,_,0));
    copy(s2r_copy_b, tXsB(_,_,0,smem_read_stage), tCrB_view(_,_,0));


    // Main Loop
    for(int itile = 0; itile < num_tile_k; ++itile) {
        for(int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;
            
            // 如果当前 Tile 的数据用完了 (ik 是最后一个 slice)，需要切换到下一个 Shared Memory Stage
            if (ik == nk - 1) {
                cp_async_wait<Config::kStage - 2>();
                __syncthreads();
                smem_read_stage = (smem_read_stage + 1) % k_stage;
            }

            // Pipeline: 加载下一个 K-slice 到寄存器 (覆盖旧数据)
            // 关键点：我们在计算当前 slice (ik) 的同时，加载下一个 slice (ik_next)
            // 如果 ik == nk-1，我们已经切换了 smem_read_stage，所以加载的是下一个 Tile 的第 0 个 slice
            copy(s2r_copy_a, tXsA(_,_,ik_next,smem_read_stage), tCrA_view(_,_,ik_next));
            copy(s2r_copy_b, tXsB(_,_,ik_next,smem_read_stage), tCrB_view(_,_,ik_next));

            // 在每个 Tile 开始时 (ik==0)，发起下一轮的 Global -> Shared 预取
            // 仅当还有剩余 Tile 需要加载时执行
            if (ik == 0 && itile < num_tile_k - (k_stage - 1)) {
                copy(gmem_copy, tAgA(_,_,_,itile + k_stage - 1), tAsA(_,_,_,smem_write_stage));
                copy(gmem_copy, tBgB(_,_,_,itile + k_stage - 1), tBsB(_,_,_,smem_write_stage));
                cp_async_fence();
                smem_write_stage = (smem_write_stage + 1) % k_stage;
            }

            // 计算当前 K-slice (数据已经在 Prologue 或上一次循环中加载好了)
            gemm(tiled_mma, tCrC, tCrA(_,_,ik), tCrB(_,_,ik), tCrC);
        }
    }

    __syncthreads();


    // reg --> smem
    T* smem_C = smem;

    Tensor sC = make_tensor(make_smem_ptr(smem_C), typename Config::SmemLayoutC{});
    auto r2s_tiled_copy_c = make_tiled_copy_C(typename Config::R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(threadIdx.x);

    // retile_S: 将 MMA 累加器视图 (tCrC) 重排为拷贝视图
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrC);
    // partition_D: 将 Shared Memory (sC) 切分以匹配拷贝操作
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);

    // 执行拷贝：现在 CuTe 知道如何将寄存器中的数据搬运到 Smem 中了
    copy(r2s_tiled_copy_c, tCrC_r2s, tCsC_r2s);


    __syncthreads();

    //smem -->  gloabl mem
    typename Config::GmemCopyC gmemcopyC;
    auto thr_copy_c = gmemcopyC.get_slice(threadIdx.x);

    auto tCsC_gmem = thr_copy_c.partition_S(sC);
    auto tCgC_gmem = thr_copy_c.partition_D(gC);

    copy(gmemcopyC,tCsC_gmem,tCgC_gmem);

}

} // namespace gemm_multi_stage