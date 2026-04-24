#pragma once

// ============================================================================
// Kernel: gemm_kernel_fp8_tma  (单 WarpGroup, TMA + WGMMA-SS, FP8 输入)
//
// 设计:
//   128 线程 (1 WarpGroup), TMA 搬运 + wgmma SS 模式计算
//   输入: E4M3 × E4M3 (或 E5M2)，FP32 累加，FP32 输出
//
// FP8 vs BF16 关键差异:
//   1. MMA 指令 K 维度: FP8 K=32 (vs BF16 K=16)，kTileK 需 32 的倍数
//   2. 推荐 kTileK=128 (= 4 × K_MMA)，每个 tile 做 4 次 wgmma 指令
//   3. SMEM 大小: FP8 每元素 1 字节，比 BF16 省一半 SMEM
//      kTileK=128: sA = 128×128×1B = 16KB/stage，kStage=3 → 48KB/tile
//   4. TMA 搬运逻辑与 BF16 完全相同（同样 1 线程发起整块）
//   5. Epilogue R->S->G 与 BF16 相同（FP32 写回）
//
// 流水线 (软件 kStage 级):
//   Prologue: 预取 min(kStage-1, num_k_tiles) 个 tile 到 SMEM
//   MainLoop:
//     wait(stage) → wgmma(stage) → wgmma_wait<1> → TMA_issue(next_k)
//
// WAR 安全性:
//   TMA 在 wgmma_wait<1> 之后发起, 保证上一轮 wgmma 已完成读取
//
// 版本说明:
//   v1/v3: 标准 TMA + wgmma SS，按 blockIdx.y/x 自然顺序
//   v_swizzle: 加入 block swizzle（log2(N_TILES_PER_GROUP) 列分组），
//              改善 L2 cache 命中率
//
// 寄存器 (kTileM=128, kTileN=128, kTileK=128):
//   163 regs, 0 spill (ptxas 实测)
// SMEM (kStage=3): ~96KB；(kStage=4): ~128KB
// ============================================================================

#include "config_fp8.cuh"
#include "mbarrier.cuh"
#include "cutlass/device_kernel.h"   // CUTLASS_GRID_CONSTANT

namespace gemm_fp8_sm90 {

// ============================================================================
// 基础版本: v1 / v3 用此模板 (自然 block 顺序)
// ============================================================================
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__global__ void __launch_bounds__(Config::kNumThreads, 1)
gemm_kernel_fp8_tma(
    float*                         __restrict__ Cptr,
    CUTLASS_GRID_CONSTANT TmaCopyA const tma_a,
    CUTLASS_GRID_CONSTANT TmaCopyB const tma_b,
    int M, int N, int K)
{
    using TA = typename Config::TA;
    using TB = typename Config::TB;
    static constexpr int kTileM = Config::kTileM;
    static constexpr int kTileN = Config::kTileN;
    static constexpr int kTileK = Config::kTileK;
    static constexpr int kStage = Config::kStage;

    // ------------------------------------------------------------------
    // 共享内存布局:
    //   [A buf: cosize(SmemLayoutA) × sizeof(TA)]
    //   [B buf: cosize(SmemLayoutB) × sizeof(TB)]
    //   [mbar[kStage]: 8B each, 8B 对齐]
    // ------------------------------------------------------------------
    extern __shared__ char smem_buf[];

    constexpr size_t smem_bytes_A = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(TA);
    constexpr size_t smem_bytes_B = cute::cosize(typename Config::SmemLayoutB{}) * sizeof(TB);
    constexpr size_t mbar_offset  = (smem_bytes_A + smem_bytes_B + 7) & ~7;

    TA*       smem_A_ptr = reinterpret_cast<TA*>(smem_buf);
    TB*       smem_B_ptr = reinterpret_cast<TB*>(smem_buf + smem_bytes_A);
    uint64_t* mbar       = reinterpret_cast<uint64_t*>(smem_buf + mbar_offset);

    auto sA = make_tensor(make_smem_ptr(smem_A_ptr), typename Config::SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(smem_B_ptr), typename Config::SmemLayoutB{});

    int tid = threadIdx.x;

    // ------------------------------------------------------------------
    // 初始化 mbarrier (arrive_count=1: 只有 tid==0 的 arrive_and_expect_tx)
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

    auto gC_full = make_tensor(make_gmem_ptr(Cptr),
                               make_shape(M, N), make_stride(N, _1{}));
    auto gC = local_tile(gC_full, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(by, bx));

    auto cta_tma_a = tma_a.get_slice(Int<0>{});
    auto cta_tma_b = tma_b.get_slice(Int<0>{});

    auto sA_pi = as_position_independent_swizzle_tensor(sA);
    auto sB_pi = as_position_independent_swizzle_tensor(sB);

    Tensor tAgA = cta_tma_a.partition_S(gA);
    Tensor tAsA = cta_tma_a.partition_D(sA_pi);
    Tensor tBgB = cta_tma_b.partition_S(gB);
    Tensor tBsB = cta_tma_b.partition_D(sB_pi);

    // ------------------------------------------------------------------
    // MMA 分区 (SS 模式: A/B 从 SMEM descriptor 读取)
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

    constexpr int kTmaBytes = kTileM * kTileK * sizeof(TA)
                            + kTileN * kTileK * sizeof(TB);
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

    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    // ------------------------------------------------------------------
    // Epilogue: R->S->G 两级写回
    // ------------------------------------------------------------------
    __syncthreads();

    using AccumType = typename Config::AccumType;
    AccumType* smem_C_ptr = reinterpret_cast<AccumType*>(smem_buf);
    auto sC = make_tensor(make_smem_ptr(smem_C_ptr), typename Config::SmemLayoutC{});

    auto r2s_copy_c = make_tiled_copy_C(typename Config::R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_c  = r2s_copy_c.get_slice(tid);
    auto tCrC_r2s   = r2s_thr_c.retile_S(tCrC);
    auto tCsC_r2s   = r2s_thr_c.partition_D(sC);
    copy(r2s_copy_c, tCrC_r2s, tCsC_r2s);

    __syncthreads();

    typename Config::S2GCopyC s2g_copy_c;
    auto s2g_thr_c = s2g_copy_c.get_slice(tid);
    auto tCsC_s2g  = s2g_thr_c.partition_S(sC);
    auto tCgC_s2g  = s2g_thr_c.partition_D(gC);
    copy(s2g_copy_c, tCsC_s2g, tCgC_s2g);
}

// ============================================================================
// v4: Block Swizzle 版本
//
// Block swizzle 将 grid 中相邻的 block 重新映射，使得同时活跃的 block
// 共享 L2 cache 中的 A/B tile，减少 DRAM bandwidth 压力。
//
// 策略: log2(kGroupSize) 列分组
//   - grid.x 方向 (N): 每 kGroupSize 个 block 为一组
//   - 在组内按列优先顺序排列，跨组按行顺序
//   - 典型 kGroupSize=8 对 H20 (108 SM) 较优
// ============================================================================
template <typename Config, typename TmaCopyA, typename TmaCopyB,
          int kGroupSize = 8>
__global__ void __launch_bounds__(Config::kNumThreads, 1)
gemm_kernel_fp8_tma_swizzle(
    float*                         __restrict__ Cptr,
    CUTLASS_GRID_CONSTANT TmaCopyA const tma_a,
    CUTLASS_GRID_CONSTANT TmaCopyB const tma_b,
    int M, int N, int K)
{
    using TA = typename Config::TA;
    using TB = typename Config::TB;
    static constexpr int kTileM = Config::kTileM;
    static constexpr int kTileN = Config::kTileN;
    static constexpr int kTileK = Config::kTileK;
    static constexpr int kStage = Config::kStage;

    extern __shared__ char smem_buf[];

    constexpr size_t smem_bytes_A = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(TA);
    constexpr size_t smem_bytes_B = cute::cosize(typename Config::SmemLayoutB{}) * sizeof(TB);
    constexpr size_t mbar_offset  = (smem_bytes_A + smem_bytes_B + 7) & ~7;

    TA*       smem_A_ptr = reinterpret_cast<TA*>(smem_buf);
    TB*       smem_B_ptr = reinterpret_cast<TB*>(smem_buf + smem_bytes_A);
    uint64_t* mbar       = reinterpret_cast<uint64_t*>(smem_buf + mbar_offset);

    auto sA = make_tensor(make_smem_ptr(smem_A_ptr), typename Config::SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(smem_B_ptr), typename Config::SmemLayoutB{});

    int tid = threadIdx.x;

    if (tid == 0) {
        for (int s = 0; s < kStage; ++s) mbar_init(&mbar[s], 1);
        mbar_fence_init();
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Block Swizzle: 将线性 block id 映射到 (bx, by)
    //
    // 原始 grid: (grid.x = N/kTileN, grid.y = M/kTileM)
    // linear_id = blockIdx.y * grid.x + blockIdx.x
    //
    // Swizzle 策略 (log2-group):
    //   num_n_tiles = N / kTileN
    //   group_id    = linear_id / kGroupSize
    //   intra_id    = linear_id % kGroupSize
    //   by_sw = group_id % (M/kTileM)
    //   bx_sw = (group_id / (M/kTileM)) * kGroupSize + intra_id
    // ------------------------------------------------------------------
    int num_m_tiles = M / kTileM;
    int num_n_tiles = N / kTileN;
    int linear_id   = blockIdx.y * num_n_tiles + blockIdx.x;
    int group_id    = linear_id / kGroupSize;
    int intra_id    = linear_id % kGroupSize;
    int by = group_id % num_m_tiles;
    int bx = (group_id / num_m_tiles) * kGroupSize + intra_id;
    // 边界保护（N 方向非整除时 bx 可能越界，此处 N 是 kTileN 的整数倍所以没问题）

    // ------------------------------------------------------------------
    // TMA 视图 & 分区
    // ------------------------------------------------------------------
    auto mA = tma_a.get_tma_tensor(make_shape(M, K));
    auto mB = tma_b.get_tma_tensor(make_shape(N, K));

    auto gA = local_tile(mA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(by, _));
    auto gB = local_tile(mB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(bx, _));

    auto gC_full = make_tensor(make_gmem_ptr(Cptr),
                               make_shape(M, N), make_stride(N, _1{}));
    auto gC = local_tile(gC_full, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                         make_coord(by, bx));

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

    constexpr int kTmaBytes = kTileM * kTileK * sizeof(TA)
                            + kTileN * kTileK * sizeof(TB);
    int num_k_tiles = K / kTileK;

    // Prologue
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

    // Main Loop
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

    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    // Epilogue
    __syncthreads();

    using AccumType = typename Config::AccumType;
    AccumType* smem_C_ptr = reinterpret_cast<AccumType*>(smem_buf);
    auto sC = make_tensor(make_smem_ptr(smem_C_ptr), typename Config::SmemLayoutC{});

    auto r2s_copy_c = make_tiled_copy_C(typename Config::R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_c  = r2s_copy_c.get_slice(tid);
    auto tCrC_r2s   = r2s_thr_c.retile_S(tCrC);
    auto tCsC_r2s   = r2s_thr_c.partition_D(sC);
    copy(r2s_copy_c, tCrC_r2s, tCsC_r2s);

    __syncthreads();

    typename Config::S2GCopyC s2g_copy_c;
    auto s2g_thr_c = s2g_copy_c.get_slice(tid);
    auto tCsC_s2g  = s2g_thr_c.partition_S(sC);
    auto tCgC_s2g  = s2g_thr_c.partition_D(gC);
    copy(s2g_copy_c, tCsC_s2g, tCgC_s2g);
}

} // namespace gemm_fp8_sm90
