#pragma once

// ============================================================================
// Kernel 3: gemm_kernel_pingpong  (Consumer WG + Producer Warp, TMA + GMMA)
//
// 设计 (v5: 隔离 Consumer/Producer 寄存器, 真正 occupancy=2):
//   160 线程:
//     Consumer WG (tid   0-127): 128 线程, 专职 wgmma 计算
//     Producer    (tid 128-159):  32 线程 (1 warp), 专职 TMA 数据搬运
//
//   两组通过两套 mbarrier 异步协作:
//     mbar_full[s]:  Producer 完成 TMA 后 signal → Consumer 等待
//     mbar_empty[s]: Consumer 完成 wgmma 后 signal → Producer 等待 (可覆盖)
//
// 演化历史:
//
//   v1 (256 线程, __noinline__):
//     将 consumer loop 提取为 __noinline__, 但 MMA fragment 仍在 kernel 本体.
//     ptxas 报告 255 reg/thread × 256 = 65280 → occupancy=1. 无改善.
//
//   v2 (256 线程, __noinline__, MMA fragment 移入子函数):
//     ptxas kernel 本体 = 40 reg, consumer 函数 = 160 reg.
//     但 C7510 警告: __noinline__ 导致 wgmma pipeline 在函数边界被序列化.
//     实测: PingPong(k3)=603us vs TMA(k3)=522us, 差距 81us(15.7%).
//
//   v3 (__forceinline__, setmaxnreg):
//     消除 C7510, wgmma pipeline 完整. 实测 PingPong(k3)=594us.
//     但 setmaxnreg 只对 persistent kernel 有效! SM 在 block 调度时
//     用 ptxas 报告值 (160 reg × 256 = 40960), 只驻留 1 block.
//     实测 PingPong(k3)=594us ≈ TMA(k4)(occupancy=1)=593us.
//     → setmaxnreg 无法提升非 persistent kernel 的 occupancy.
//
//   v4 (__forceinline__, 160 线程):
//     核心洞察: 减少 block 总线程数. 但实测 ptxas 报告 232 reg!
//     原因: __forceinline__ 把 Producer(TMA 视图) + Consumer(wgmma 累加器)
//     全部内联进 kernel 本体. ptxas 为 if/else 两个分支都分配寄存器.
//       232 reg × 160 = 37120 reg/block → floor(65536/37120) = 1 block ❌
//     性能: PingPong(k3)=590us ≈ TMA(k4)(occupancy=1), 并未提升.
//
//   v5 (当前, 160 线程, Consumer __forceinline__ + Producer __noinline__):
//     关键洞察:
//       - Consumer 执行 wgmma → 必须 __forceinline__ 避免 C7510
//       - Producer 只做 TMA → 可以 __noinline__, 不影响 wgmma pipeline
//       - __noinline__ Producer 将 TMA 视图寄存器隔离在独立栈帧中
//       - kernel 本体对 Consumer 线程: 只有 Consumer 相关变量 (gC, sA, sB)
//       - TMA 视图 (tAgA, tAsA, tBgB, tBsB) 移入 Producer 函数内部构建
//     预期:
//       kernel 本体寄存器 ≈ Consumer 寄存器 (~154 reg)
//       154 reg × 160 = 24640 reg/block → floor(65536/24640) = 2 blocks ✅
//       kStage=3: SMEM=96KB → floor(227/96) = 2 blocks → occupancy=2 ✅
//
// SMEM: 128 KB (kStage=4) / 96 KB (kStage=3)
// ============================================================================

#include "detail/config.cuh"
#include "detail/mbarrier.cuh"
#include "cutlass/device_kernel.h"   // CUTLASS_GRID_CONSTANT

namespace gemm_sm90 {

using namespace cute;

// ============================================================================
// pp_producer_loop  —  Producer (1 warp, tid 128-159) 主循环
//
// v5 关键改变: 改为 __noinline__
//   - Producer 不执行 wgmma, 无 C7510 风险
//   - __noinline__ 将 TMA 张量视图 (tAgA/tAsA/tBgB/tBsB) 的寄存器
//     限制在本函数的独立栈帧中, 不污染 kernel 本体的寄存器预算
//   - 接受原始指针和坐标, 在函数内部重建 TMA 视图
//   - 只有 prod_tid==0 (即 threadIdx.x==128) 执行 TMA 发起和 mbar 操作
// ============================================================================
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__device__ __noinline__ void pp_producer_loop(
    uint64_t* __restrict__ mbar_full,
    uint64_t* __restrict__ mbar_empty,
    TmaCopyA const& tma_a,
    TmaCopyB const& tma_b,
    char* __restrict__ smem_A_ptr,   // raw SMEM 指针, 在函数内重建 tensor
    char* __restrict__ smem_B_ptr,
    int prod_tid,      // tid - 128, 范围 [0, 32)
    int bx, int by,
    int M, int N, int K)
{
    static constexpr int kStage    = Config::kStage;
    static constexpr int kTileM    = Config::kTileM;
    static constexpr int kTileN    = Config::kTileN;
    static constexpr int kTileK    = Config::kTileK;
    using T = typename Config::T;
    static constexpr int kTmaBytes = kTileM * kTileK * sizeof(T)
                                   + kTileN * kTileK * sizeof(T);

    // 只有 prod_tid==0 执行 TMA 操作
    if (prod_tid == 0) {
        // 在函数内部构建 TMA 视图 (寄存器限制在本函数栈帧)
        auto mA = tma_a.get_tma_tensor(make_shape(M, K));
        auto mB = tma_b.get_tma_tensor(make_shape(N, K));

        auto gA = local_tile(mA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(by, _));
        auto gB = local_tile(mB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(bx, _));

        auto sA = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_A_ptr)),
                              typename Config::SmemLayoutA{});
        auto sB = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_B_ptr)),
                              typename Config::SmemLayoutB{});

        auto cta_tma_a = tma_a.get_slice(Int<0>{});
        auto cta_tma_b = tma_b.get_slice(Int<0>{});

        auto sA_pi = as_position_independent_swizzle_tensor(sA);
        auto sB_pi = as_position_independent_swizzle_tensor(sB);

        Tensor tAgA = cta_tma_a.partition_S(gA);
        Tensor tAsA = cta_tma_a.partition_D(sA_pi);
        Tensor tBgB = cta_tma_b.partition_S(gB);
        Tensor tBsB = cta_tma_b.partition_D(sB_pi);

        int num_k_tiles = K / kTileK;
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
}

// ============================================================================
// pp_consumer_loop  —  Consumer WarpGroup 主循环 + Epilogue
//
// Consumer WG: 全 128 线程 (tid 0-127), 整个 WarpGroup 参与 wgmma.
//
// v5: 保持 __forceinline__ (避免 C7510: wgmma pipeline 跨函数边界)
//   wg_tid = threadIdx.x (0-127)
//
// 关键: 全 128 线程必须都执行 mbar_wait(&mbar_full[stage], phase_f)
//   acquire 语义只对"执行该指令的线程"有效.
// ============================================================================
template <typename Config, typename SmemTensorA, typename SmemTensorB,
          typename GmemTensorC>
__device__ __forceinline__ void pp_consumer_loop(
    uint64_t* __restrict__ mbar_full,
    uint64_t* __restrict__ mbar_empty,
    SmemTensorA const& sA,
    SmemTensorB const& sB,
    GmemTensorC const& gC,
    int wg_tid,
    int num_k_tiles)
{
    static constexpr int kStage = Config::kStage;

    // ------------------------------------------------------------------
    // 在 Consumer 函数内部构建 MMA 视图
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
    // 主循环: 软件流水线 (kStage 级)
    //
    // C7520 消除: 用 mbar_arrive_if (PTX @pred) 替代 if (wg_tid==0) mbar_arrive,
    //   消除 divergent path, 避免编译器插入 WG.AR 序列化 wgmma.
    // ------------------------------------------------------------------
    int stage   = 0;
    int phase_f = 0;

    for (int k = 0; k < num_k_tiles; ++k) {
        // 全 128 线程等待 TMA 写完 stage (acquire 语义保证内存可见)
        mbar_wait(&mbar_full[stage], phase_f);

        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        gemm(tiled_mma,
             tCrA(_, _, _, stage),
             tCrB(_, _, _, stage),
             tCrC);
        warpgroup_commit_batch();

        // 等上一轮 wgmma 完成; 上一轮完成后 prev_stage SMEM 安全可覆盖
        warpgroup_wait<1>();
        warpgroup_fence_operand(tCrC);

        // 通知 Producer: prev_stage 的 SMEM 已用完
        // mbar_arrive_if 用 PTX @pred: k==0 时 no-op, wg_tid!=0 时也 no-op
        int prev_stage = (stage - 1 + kStage) % kStage;
        mbar_arrive_if(&mbar_empty[prev_stage], (k > 0) && (wg_tid == 0));

        stage = (stage + 1) % kStage;
        if (stage == 0) phase_f ^= 1;
    }

    // Drain: 等最后一轮 wgmma 完成
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    // 释放最后一个 stage
    int last_stage = (stage - 1 + kStage) % kStage;
    mbar_arrive_if(&mbar_empty[last_stage], (num_k_tiles > 0) && (wg_tid == 0));

    // ------------------------------------------------------------------
    // Epilogue: 写回 gmem
    // ------------------------------------------------------------------
    copy(tCrC, tCgC);
}

// ============================================================================
// gemm_kernel_pingpong  —  kernel 本体 (v5: 隔离 Consumer/Producer 寄存器)
//
// 线程布局:
//   tid  0-127: Consumer WarpGroup (128 线程, 完整 WarpGroup)
//   tid 128-159: Producer Warp    ( 32 线程, 1 warp)
//
// Occupancy 分析 (kStage=3):
//   v5 关键: TMA 张量视图移入 __noinline__ Producer 函数内部
//   kernel 本体对 Consumer 线程只有 gC/sA/sB 及 wgmma 累加器
//   预期 ptxas: ~154 reg/thread (wgmma 累加器主导, 无 TMA 视图寄存器)
//   寄存器占用: 154 × 160 = 24640 reg/block
//   SM 可驻留: floor(65536/24640) = 2 blocks
//   SMEM 限制: floor(227KB/96KB) = 2 blocks
//   → 实际 occupancy = 2 ✅
//
// __launch_bounds__(160, 2):
//   160 线程/block; min_blocks_per_sm=2 (提示 ptxas 目标 occupancy=2)
//   若寄存器超出 floor(65536/(2×160))=204, ptxas 会优先 spill (可接受).
// ============================================================================
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__global__ void __launch_bounds__(Config::kNumThreadsPP, 2)
gemm_kernel_pingpong(
    float*       __restrict__ Cptr,
    CUTLASS_GRID_CONSTANT TmaCopyA const tma_a,
    CUTLASS_GRID_CONSTANT TmaCopyB const tma_b,
    int M, int N, int K)
{
    using T = typename Config::T;
    static constexpr int kTileM = Config::kTileM;
    static constexpr int kTileN = Config::kTileN;
    static constexpr int kStage = Config::kStage;

    // ------------------------------------------------------------------
    // 共享内存布局
    // ------------------------------------------------------------------
    extern __shared__ char smem_buf[];

    constexpr size_t smem_bytes_A = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(T);
    constexpr size_t smem_bytes_B = cute::cosize(typename Config::SmemLayoutB{}) * sizeof(T);
    constexpr size_t mbar_offset  = (smem_bytes_A + smem_bytes_B + 7) & ~7;

    char*     smem_A_ptr = smem_buf;
    char*     smem_B_ptr = smem_buf + smem_bytes_A;
    uint64_t* mbar_full  = reinterpret_cast<uint64_t*>(smem_buf + mbar_offset);
    uint64_t* mbar_empty = mbar_full + kStage;

    int tid            = threadIdx.x;
    const bool is_producer = (tid >= 128);  // tid 128-159: Producer warp
    const int  wg_tid      = tid;           // Consumer: tid 0-127 直接就是 wg_tid
    const int  prod_tid    = tid - 128;     // Producer: 0-31

    // ------------------------------------------------------------------
    // mbarrier 初始化 (tid==0 执行)
    // ------------------------------------------------------------------
    if (tid == 0) {
        for (int s = 0; s < kStage; ++s) {
            mbar_init(&mbar_full[s],  1);
            mbar_init(&mbar_empty[s], 1);
        }
        mbar_fence_init();
    }
    __syncthreads();

    // Consumer (wg_tid==0) 预先 arrive mbar_empty, 让 Producer 一开始无需等待
    if (!is_producer && wg_tid == 0) {
        for (int s = 0; s < kStage; ++s) {
            mbar_arrive(&mbar_empty[s]);
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // 分发 (Consumer/Producer 各走独立代码路径)
    //
    // 关键: Consumer 路径下没有 TMA 相关变量 (tma_a/b 的 tensor view)
    //   gC, sA, sB 在 Consumer 路径内构建, TMA 视图在 Producer 函数内构建
    //   ptxas 分析 Consumer 线程时只看到 wgmma 相关寄存器 (~154 reg)
    // ------------------------------------------------------------------
    if (is_producer) {
        // Producer: TMA 张量视图在函数内构建 (__noinline__ 隔离寄存器)
        pp_producer_loop<Config>(
            mbar_full, mbar_empty,
            tma_a, tma_b,
            smem_A_ptr, smem_B_ptr,
            prod_tid,
            blockIdx.x, blockIdx.y,
            M, N, K);
    } else {
        // Consumer: 构建 gC / sA / sB (无 TMA 视图寄存器)
        auto sA = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_A_ptr)),
                              typename Config::SmemLayoutA{});
        auto sB = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_B_ptr)),
                              typename Config::SmemLayoutB{});
        auto C  = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(N, _1{}));
        auto gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                             make_coord(blockIdx.y, blockIdx.x));

        int num_k_tiles = K / Config::kTileK;
        pp_consumer_loop<Config>(
            mbar_full, mbar_empty,
            sA, sB, gC,
            wg_tid, num_k_tiles);
    }

    __syncthreads();
}

} // namespace gemm_sm90
