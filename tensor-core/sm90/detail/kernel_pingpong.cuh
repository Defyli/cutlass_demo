#pragma once

// ============================================================================
// Kernel 3: gemm_kernel_pingpong  (双 WarpGroup Ping-Pong, TMA + GMMA)
//
// 设计:
//   256 线程 (2 WarpGroup):
//     WG1 (tid 128-255): Producer — 专职 TMA 数据搬运
//     WG0 (tid   0-127): Consumer — 专职 wgmma 计算
//
//   两组通过两套 mbarrier 异步协作:
//     mbar_full[s]:  Producer 完成 TMA 后 signal → Consumer 等待
//     mbar_empty[s]: Consumer 完成 wgmma 后 signal → Producer 等待 (可覆盖)
//
// 流水线时序 (kStage=4 为例):
//
//     Producer (WG1)                Consumer (WG0)
//     ──────────────────────────    ──────────────────────────────────
//     wait_empty(s0) → TMA k=0→s0
//     wait_empty(s1) → TMA k=1→s1
//     wait_empty(s2) → TMA k=2→s2
//     wait_empty(s3) → TMA k=3→s3  wait_full(s0) → wgmma(s0)
//     wait_empty(s0) ...            wait_full(s1) → wgmma(s1) → signal_empty(s0)
//     ...                           ...
//
// 寄存器分离方案 (__noinline__ 子函数):
//   同一 kernel 体内 Producer + Consumer 共享编译单元时,
//   ptxas 对整个 kernel 取寄存器上限 max(~40, ~154) 实际输出 232,
//   导致 232×256 = 59392 reg/block, SM 仅能驻留 1 个 block.
//
//   解决: 将两路分别抽成 __noinline__ __device__ 子函数.
//   ptxas 独立优化每个子函数:
//     pp_producer_loop: ~40 reg  (只有循环变量 + mbar 地址)
//     pp_consumer_loop: ~154 reg (wgmma 累加器)
//   kernel 层面: max(40, 154) = 154 reg/thread × 256 threads
//              = 39424 reg/block → SM 可驻留 ~2-3 个 block
//
// SMEM: 128 KB (kStage=4) / 96 KB (kStage=3)
// ============================================================================

#include "detail/config.cuh"
#include "detail/mbarrier.cuh"
#include "cutlass/device_kernel.h"   // CUTLASS_GRID_CONSTANT

namespace gemm_sm90 {

// ============================================================================
// pp_producer_loop  —  Producer WarpGroup 主循环
//
// 此函数独立编译 (ptxas 单独分配寄存器):
//   - wg_tid==0 执行 TMA 循环, 用变量约 ~20 个
//   - 其余 127 线程仅执行 setmaxnreg.dec, 几乎不占寄存器
//   - 整个函数峰值约 40 reg
//
// setmaxnreg.dec.sync.aligned.u32 40:
//   将 Producer WG 的运行时寄存器配额降至 40, 释放多余寄存器给 SM 上的其他 block.
//   这是 warpgroup 级同步指令, 必须全 128 线程执行.
// ============================================================================
template <typename Config, typename TmaCopyA, typename TmaCopyB,
          typename TensorAgA, typename TensorAsA,
          typename TensorBgB, typename TensorBsB>
__device__ __noinline__ void pp_producer_loop(
    uint64_t* __restrict__ mbar_full,
    uint64_t* __restrict__ mbar_empty,
    TmaCopyA const& tma_a,
    TmaCopyB const& tma_b,
    TensorAgA const& tAgA,
    TensorAsA const& tAsA,
    TensorBgB const& tBgB,
    TensorBsB const& tBsB,
    int wg_tid,
    int num_k_tiles)
{
    static constexpr int kStage    = Config::kStage;
    static constexpr int kTileM    = Config::kTileM;
    static constexpr int kTileN    = Config::kTileN;
    static constexpr int kTileK    = Config::kTileK;
    using T = typename Config::T;
    static constexpr int kTmaBytes = kTileM * kTileK * sizeof(T)
                                   + kTileN * kTileK * sizeof(T);

    // 降低整个 Producer WG 的寄存器配额, 释放给 SM 上其他 block
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" :::);

    if (wg_tid == 0) {
        int stage   = 0;
        int phase_e = 0;

        for (int k = 0; k < num_k_tiles; ++k) {
            mbar_wait(&mbar_empty[stage], phase_e);

            mbar_arrive_and_expect_tx(&mbar_full[stage], kTmaBytes);
            copy(tma_a.with(mbar_full[stage]), tAgA(_, _, _, k), tAsA(_, _, _, stage));
            copy(tma_b.with(mbar_full[stage]), tBgB(_, _, _, k), tBsB(_, _, _, stage));

            stage = (stage + 1) % kStage;
            if (stage == 0) phase_e ^= 1;
        }
    }
    // 其余 127 Producer 线程 idle, 等 kernel 层面的 __syncthreads()
}

// ============================================================================
// pp_consumer_loop  —  Consumer WarpGroup 主循环
//
// 此函数独立编译 (ptxas 单独分配寄存器):
//   - 持有 wgmma 的 FP32 累加器 tCrC (~128 FP32 = 128 reg)
//   - 加上循环控制变量 ~26 reg
//   - 总计约 154 reg, 与单WG TMA kernel 持平
//
// 关键: mbar_full 的 wait 必须由全 128 线程执行!
//   mbarrier.try_wait.acquire 的内存可见性只对"执行该指令的线程"有效.
//   若只让 wg_tid==0 执行 wait, 其余 127 线程无法获得 TMA 写入的 acquire 保证,
//   会读到过时的 L1 缓存数据.
// ============================================================================
template <typename Config, typename TiledMMA,
          typename TensorCsA, typename TensorCsB, typename TensorCgC,
          typename TensorCrA, typename TensorCrB, typename TensorCrC>
__device__ __noinline__ void pp_consumer_loop(
    uint64_t* __restrict__ mbar_full,
    uint64_t* __restrict__ mbar_empty,
    TiledMMA const& tiled_mma,
    TensorCsA const&, TensorCsB const&,   // 仅占位, stage 索引由本函数管理
    TensorCgC const& tCgC,
    TensorCrA const& tCrA,
    TensorCrB const& tCrB,
    TensorCrC      & tCrC,
    int wg_tid,
    int num_k_tiles)
{
    static constexpr int kStage = Config::kStage;

    int prev_stage = -1;
    int stage      = 0;
    int phase_f    = 0;

    for (int k = 0; k < num_k_tiles; ++k) {
        // 全 128 线程等待 TMA 完成 (acquire 保证对所有执行线程可见)
        mbar_wait(&mbar_full[stage], phase_f);

        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        gemm(tiled_mma,
             tCrA(_, _, _, stage),
             tCrB(_, _, _, stage),
             tCrC);
        warpgroup_commit_batch();

        // 等上一轮 wgmma 完成 (depth=1), 上一个 stage 的 SMEM 可以覆盖
        warpgroup_wait<1>();
        warpgroup_fence_operand(tCrC);

        if (prev_stage >= 0 && wg_tid == 0) {
            mbar_arrive(&mbar_empty[prev_stage]);
        }

        prev_stage = stage;
        stage = (stage + 1) % kStage;
        if (stage == 0) phase_f ^= 1;
    }

    // Drain: 等最后一轮 wgmma 完成
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    if (prev_stage >= 0 && wg_tid == 0) {
        mbar_arrive(&mbar_empty[prev_stage]);
    }
}

// ============================================================================
// gemm_kernel_pingpong  —  kernel 本体
//
// 只负责:
//   1. SMEM 布局解析
//   2. mbarrier 初始化
//   3. TMA / MMA 视图构建
//   4. 分发到 pp_producer_loop / pp_consumer_loop (__noinline__)
//   5. Epilogue: Consumer 写回 gmem
//
// 由于两路通过 __noinline__ 调用, ptxas 统计整个 kernel 的寄存器取
//   max(pp_producer_loop, pp_consumer_loop) ≈ 154
// 而非重构前的 232.
// ============================================================================
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__global__ void __launch_bounds__(Config::kNumThreadsPP, 1)
gemm_kernel_pingpong(
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
    //   [A buf: (kTileM, kTileK, kStage)]
    //   [B buf: (kTileN, kTileK, kStage)]
    //   [mbar_full[kStage]:  Producer → Consumer, "数据就绪"]
    //   [mbar_empty[kStage]: Consumer → Producer, "SMEM 空闲"]
    // ------------------------------------------------------------------
    extern __shared__ char smem_buf[];

    constexpr size_t smem_bytes_A = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(T);
    constexpr size_t smem_bytes_B = cute::cosize(typename Config::SmemLayoutB{}) * sizeof(T);
    constexpr size_t mbar_offset  = (smem_bytes_A + smem_bytes_B + 7) & ~7;

    T*        smem_A_ptr = reinterpret_cast<T*>(smem_buf);
    T*        smem_B_ptr = reinterpret_cast<T*>(smem_buf + smem_bytes_A);
    uint64_t* mbar_full  = reinterpret_cast<uint64_t*>(smem_buf + mbar_offset);
    uint64_t* mbar_empty = mbar_full + kStage;

    auto sA = make_tensor(make_smem_ptr(smem_A_ptr), typename Config::SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(smem_B_ptr), typename Config::SmemLayoutB{});

    int tid            = threadIdx.x;
    const bool is_producer = (tid >= 128);
    const int  wg_tid      = tid % 128;

    // ------------------------------------------------------------------
    // mbarrier 初始化 (arrive_count=1: 各自只需 1 次 arrive 翻转 phase)
    // mbar_empty 预先 arrive, 让 Producer 一开始无需等待即可发起 TMA
    // ------------------------------------------------------------------
    if (tid == 0) {
        for (int s = 0; s < kStage; ++s) {
            mbar_init(&mbar_full[s],  1);
            mbar_init(&mbar_empty[s], 1);
        }
        mbar_fence_init();
    }
    __syncthreads();

    if (!is_producer && wg_tid == 0) {
        for (int s = 0; s < kStage; ++s) {
            mbar_arrive(&mbar_empty[s]);
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // TMA 视图构建
    // ------------------------------------------------------------------
    auto mA = tma_a.get_tma_tensor(make_shape(M, K));
    auto mB = tma_b.get_tma_tensor(make_shape(N, K));

    int bx = blockIdx.x;
    int by = blockIdx.y;

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

    int num_k_tiles = K / kTileK;

    // ------------------------------------------------------------------
    // MMA 视图构建 (Consumer 路径使用)
    // Producer 路径在 __noinline__ 子函数内不访问这些变量,
    // ptxas 可从 Producer 的寄存器分配中消除它们.
    // ------------------------------------------------------------------
    typename Config::TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(wg_tid);

    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    // ------------------------------------------------------------------
    // 分发到 __noinline__ 子函数 (寄存器分离的关键)
    // ------------------------------------------------------------------
    if (is_producer) {
        pp_producer_loop<Config>(
            mbar_full, mbar_empty,
            tma_a, tma_b,
            tAgA, tAsA, tBgB, tBsB,
            wg_tid, num_k_tiles);
    } else {
        pp_consumer_loop<Config>(
            mbar_full, mbar_empty,
            tiled_mma,
            tCsA, tCsB, tCgC,
            tCrA, tCrB, tCrC,
            wg_tid, num_k_tiles);
    }

    // ------------------------------------------------------------------
    // Epilogue: 只有 Consumer 写回 gmem
    // ------------------------------------------------------------------
    __syncthreads();

    if (!is_producer) {
        copy(tCrC, tCgC);
    }
}

} // namespace gemm_sm90
