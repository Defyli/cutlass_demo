#pragma once

// ============================================================================
// Kernel 1: gemm_kernel_tma  (单 WarpGroup, TMA + GMMA)
//
// 设计:
//   128 线程 (1 WarpGroup), TMA 搬运 + wgmma SS 模式计算
//
// 流水线 (软件 kStage 级):
//   Prologue: 预取 min(kStage-1, num_k_tiles) 个 tile 到 SMEM
//   MainLoop:
//     wait(stage)    → wgmma(stage) → wgmma_wait<1>
//                                   → TMA_issue(next_k → write_stage)
//
// WAR 安全性:
//   TMA 在 wgmma_wait<1> 之后发起, 保证上一轮 wgmma 已完成读取 write_stage,
//   TMA 不会覆盖仍在被 wgmma 读取的 SMEM.
//
// 寄存器: ~154 (ptxas 实测)
// SMEM:   96 KB (kStage=3) / 128 KB (kStage=4)
// ============================================================================

#include "detail/config.cuh"
#include "detail/mbarrier.cuh"
#include "cutlass/device_kernel.h"   // CUTLASS_GRID_CONSTANT

namespace gemm_sm90 {

template <typename Config, typename TmaCopyA, typename TmaCopyB>
__global__ void __launch_bounds__(Config::kNumThreads, 1)
gemm_kernel_tma(
    float*       __restrict__ Cptr,
    CUTLASS_GRID_CONSTANT TmaCopyA const tma_a,
    CUTLASS_GRID_CONSTANT TmaCopyB const tma_b,
    int M, int N, int K)
{
    using T = typename Config::T;
    static constexpr int kTileM = Config::kTileM;
    static constexpr int kTileN = Config::kTileN;
    static constexpr int kTileK = Config::kTileK;
    static constexpr int kStage = Config::kStage;

    // ------------------------------------------------------------------
    // 共享内存布局:
    //   [A buf: cosize(SmemLayoutA) * sizeof(T)]
    //   [B buf: cosize(SmemLayoutB) * sizeof(T)]
    //   [mbar[kStage]: 8B each, 8B 对齐]
    // ------------------------------------------------------------------
    extern __shared__ char smem_buf[];

    constexpr size_t smem_bytes_A = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(T);
    constexpr size_t smem_bytes_B = cute::cosize(typename Config::SmemLayoutB{}) * sizeof(T);
    constexpr size_t mbar_offset  = (smem_bytes_A + smem_bytes_B + 7) & ~7;

    T*        smem_A_ptr = reinterpret_cast<T*>(smem_buf);
    T*        smem_B_ptr = reinterpret_cast<T*>(smem_buf + smem_bytes_A);
    uint64_t* mbar       = reinterpret_cast<uint64_t*>(smem_buf + mbar_offset);

    auto sA = make_tensor(make_smem_ptr(smem_A_ptr), typename Config::SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(smem_B_ptr), typename Config::SmemLayoutB{});

    int tid = threadIdx.x;

    // ------------------------------------------------------------------
    // 初始化 mbarrier (arrive_count=1: 只有 tid==0 的 arrive_and_expect_tx 算)
    // ------------------------------------------------------------------
    if (tid == 0) {
        for (int s = 0; s < kStage; ++s) {
            mbar_init(&mbar[s], 1);
        }
        mbar_fence_init();
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // TMA 视图 & 分区
    // ------------------------------------------------------------------
    auto mA = tma_a.get_tma_tensor(make_shape(M, K));
    auto mB = tma_b.get_tma_tensor(make_shape(N, K));

    int bx = blockIdx.x;  // N 方向
    int by = blockIdx.y;  // M 方向

    auto gA = local_tile(mA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(by, _));
    auto gB = local_tile(mB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(bx, _));

    auto C  = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(N, _1{}));
    auto gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(by, bx));

    auto cta_tma_a = tma_a.get_slice(Int<0>{});
    auto cta_tma_b = tma_b.get_slice(Int<0>{});

    auto sA_pi = as_position_independent_swizzle_tensor(sA);
    auto sB_pi = as_position_independent_swizzle_tensor(sB);

    Tensor tAgA = cta_tma_a.partition_S(gA);
    Tensor tAsA = cta_tma_a.partition_D(sA_pi);
    Tensor tBgB = cta_tma_b.partition_S(gB);
    Tensor tBsB = cta_tma_b.partition_D(sB_pi);

    // ------------------------------------------------------------------
    // MMA 分区
    // ------------------------------------------------------------------
    typename Config::TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    constexpr int kTmaBytes = kTileM * kTileK * sizeof(T) + kTileN * kTileK * sizeof(T);
    int num_k_tiles = K / kTileK;

    // ------------------------------------------------------------------
    // Prologue: 预取前 kStage-1 个 tile
    // ------------------------------------------------------------------
    int write_stage    = 0;
    int prefetch_count = cute::min(kStage - 1, num_k_tiles);

    for (int s = 0; s < prefetch_count; ++s) {
        if (tid == 0) {
            mbar_arrive_and_expect_tx(&mbar[write_stage], kTmaBytes);
            copy(tma_a.with(mbar[write_stage]), tAgA(_, _, _, s), tAsA(_, _, _, write_stage));
            copy(tma_b.with(mbar[write_stage]), tBgB(_, _, _, s), tBsB(_, _, _, write_stage));
        }
        write_stage = (write_stage + 1) % kStage;
    }

    // ------------------------------------------------------------------
    // Main Loop: TMA 与 wgmma 软件流水线
    //
    // 时序 (kStage=3 为例):
    //   k=0: wait(s0) → wgmma(s0) → wait<1> → TMA k=2→s2
    //   k=1: wait(s1) → wgmma(s1) → wait<1> → TMA k=3→s0  (s0 已被 wgmma_0 读完)
    //   k=2: wait(s2) → wgmma(s2) → wait<1> → ...
    //
    // WAR 安全: TMA 在 wait<1> 之后发起, 保证 write_stage 对应 SMEM 已被上一轮
    //           wgmma 读完, 不会产生写-后-读竞争.
    // ------------------------------------------------------------------
    int read_stage = 0;
    int phase      = 0;

    for (int k = 0; k < num_k_tiles; ++k) {
        mbar_wait(&mbar[read_stage], phase);

        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        gemm(tiled_mma,
             tCrA(_, _, _, read_stage),
             tCrB(_, _, _, read_stage),
             tCrC);
        warpgroup_commit_batch();

        warpgroup_wait<1>();
        warpgroup_fence_operand(tCrC);

        read_stage = (read_stage + 1) % kStage;
        if (read_stage == 0) phase ^= 1;

        int next_k = k + (kStage - 1);
        if (next_k < num_k_tiles) {
            if (tid == 0) {
                mbar_arrive_and_expect_tx(&mbar[write_stage], kTmaBytes);
                copy(tma_a.with(mbar[write_stage]),
                     tAgA(_, _, _, next_k), tAsA(_, _, _, write_stage));
                copy(tma_b.with(mbar[write_stage]),
                     tBgB(_, _, _, next_k), tBsB(_, _, _, write_stage));
            }
            write_stage = (write_stage + 1) % kStage;
        }
    }

    // Drain
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    __syncthreads();
    copy(tCrC, tCgC);
}

} // namespace gemm_sm90
