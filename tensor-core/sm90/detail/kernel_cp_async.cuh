#pragma once

// ============================================================================
// Kernel 2: gemm_kernel_cp_async  (单 WarpGroup, cp.async + GMMA)
//
// 设计:
//   128 线程 (1 WarpGroup), cp.async 搬运 (Ada 风格) + wgmma SS 模式计算
//
// 与 gemm_kernel_tma 的差异:
//   - G->S: cp.async (128线程分工) 而非 TMA (1线程发起)
//   - 同步: cp_async_fence + cp_async_wait<N> + __syncthreads()
//           而非 mbarrier (phase-based)
//
// 用途: 隔离对比 TMA vs cp.async 的数据搬运性能差异
// 寄存器: ~198
// SMEM:   96 KB (kStage=3)
// ============================================================================

#include "detail/config.cuh"

namespace gemm_sm90 {

template <typename Config>
__global__ void __launch_bounds__(Config::kNumThreads, 1)
gemm_kernel_cp_async(
    float*       __restrict__ Cptr,
    const typename Config::T* __restrict__ Aptr,
    const typename Config::T* __restrict__ Bptr,
    int M, int N, int K)
{
    using T = typename Config::T;
    static constexpr int kTileM = Config::kTileM;
    static constexpr int kTileN = Config::kTileN;
    static constexpr int kTileK = Config::kTileK;
    static constexpr int kStage = Config::kStage;

    extern __shared__ char smem_buf[];
    constexpr size_t smem_bytes_A = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(T);

    T* smem_A_ptr = reinterpret_cast<T*>(smem_buf);
    T* smem_B_ptr = reinterpret_cast<T*>(smem_buf + smem_bytes_A);

    auto sA = make_tensor(make_smem_ptr(smem_A_ptr), typename Config::SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(smem_B_ptr), typename Config::SmemLayoutB{});

    int tid = threadIdx.x;
    int bx  = blockIdx.x;
    int by  = blockIdx.y;

    // gmem tensors
    auto gA_full = make_tensor(make_gmem_ptr(Aptr), make_shape(M, K), make_stride(K, _1{}));
    auto gB_full = make_tensor(make_gmem_ptr(Bptr), make_shape(N, K), make_stride(K, _1{}));
    auto gC_full = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(N, _1{}));

    auto gA = local_tile(gA_full, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(by, _));
    auto gB = local_tile(gB_full, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(bx, _));
    auto gC = local_tile(gC_full, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(by, bx));

    // cp.async copy setup
    typename Config::G2SCopyA g2s_copy_a;
    typename Config::G2SCopyB g2s_copy_b;
    auto g2s_thr_a = g2s_copy_a.get_slice(tid);
    auto g2s_thr_b = g2s_copy_b.get_slice(tid);

    auto tAgA = g2s_thr_a.partition_S(gA);
    auto tAsA = g2s_thr_a.partition_D(sA);
    auto tBgB = g2s_thr_b.partition_S(gB);
    auto tBsB = g2s_thr_b.partition_D(sB);

    // MMA setup
    typename Config::TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    int num_k_tiles = K / kTileK;

    // Prologue: 预取 kStage-1 个 tile
    int prefetch = cute::min(kStage - 1, num_k_tiles);
    for (int s = 0; s < prefetch; ++s) {
        copy(g2s_copy_a, tAgA(_, _, _, s), tAsA(_, _, _, s));
        copy(g2s_copy_b, tBgB(_, _, _, s), tBsB(_, _, _, s));
        cp_async_fence();
    }
    cp_async_wait<Config::kStage - 2>();
    __syncthreads();

    int read_stage  = 0;
    int write_stage = prefetch % kStage;

    for (int k = 0; k < num_k_tiles; ++k) {
        int next_k = k + kStage - 1;
        if (next_k < num_k_tiles) {
            copy(g2s_copy_a, tAgA(_, _, _, next_k), tAsA(_, _, _, write_stage));
            copy(g2s_copy_b, tBgB(_, _, _, next_k), tBsB(_, _, _, write_stage));
            cp_async_fence();
            write_stage = (write_stage + 1) % kStage;
        }

        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        gemm(tiled_mma,
             tCrA(_, _, _, read_stage),
             tCrB(_, _, _, read_stage),
             tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrC);

        cp_async_wait<Config::kStage - 2>();
        __syncthreads();

        read_stage = (read_stage + 1) % kStage;
    }

    __syncthreads();
    copy(tCrC, tCgC);
}

} // namespace gemm_sm90
