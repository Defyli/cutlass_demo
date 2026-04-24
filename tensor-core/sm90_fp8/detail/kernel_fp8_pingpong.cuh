#pragma once

// ============================================================================
// FP8 Ping-Pong Kernel (v5 寄存器隔离模式)
//
// 设计 (参考 BF16 kernel_pingpong.cuh v5):
//   160 线程:
//     Consumer WG  (tid   0-127): 128 线程, 专职 wgmma FP8 计算
//     Producer     (tid 128-159):  32 线程 (1 warp), 专职 TMA 数据搬运
//
//   Producer 使用 __noinline__ 隔离 TMA 视图寄存器
//   Consumer 使用 __forceinline__ 避免 C7510 wgmma 流水线序列化
//
// FP8 与 BF16 差异:
//   1. 输入类型 TA/TB (1字节) 而非单一的 T (2字节)
//   2. wgmma K=32 (FP8) vs K=16 (BF16)
//   3. 累加器 FP32，Epilogue 需要 R->S->G (SMEM 中转)
//      → BF16 ping-pong 可以 copy(tCrC, tCgC) 直接 R->G
//      → FP8 同样可以 copy(tCrC, tCgC) (FP32 acc 直接写 FP32 gmem)，更简单
//      → 但需要确保 write address 是 16B 对齐 (128bit/thread)
//
// 寄存器分析 (kTileM=128, kTileN=128, kTileK=128, kStage=3):
//   Consumer 主导: FP8 wgmma FP32 acc = kTileM/WG × kTileN = 32 × 128 = 4096 FP32 = 128 regs
//   BF16 类似: 实测 ~154 regs
//   FP8 预期:  ~163 regs (实测 v1)
//   寄存器占用: 163 × 160 = 26080 reg/block → floor(65536/26080) = 2 blocks ✅
//   SMEM 限制: 96KB × 2 = 192KB < 227KB → 2 blocks ✅
//   → 理论 occupancy = 2
//
// SMEM (kStage=3): 96KB (AB) + epilogue sC 复用
// ============================================================================

#include "config_fp8.cuh"
#include "mbarrier.cuh"
#include "cutlass/device_kernel.h"   // CUTLASS_GRID_CONSTANT

namespace gemm_fp8_sm90 {

using namespace cute;

// ============================================================================
// pp_fp8_producer_loop  —  Producer (1 warp, tid 128-159) 主循环
//
// __noinline__: TMA 视图寄存器限制在本函数栈帧，不污染 Consumer 的寄存器预算
// ============================================================================
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__device__ __noinline__ void pp_fp8_producer_loop(
    uint64_t* __restrict__ mbar_full,
    uint64_t* __restrict__ mbar_empty,
    TmaCopyA const& tma_a,
    TmaCopyB const& tma_b,
    char* __restrict__ smem_A_ptr,
    char* __restrict__ smem_B_ptr,
    int prod_tid,
    int bx, int by,
    int M, int N, int K)
{
    using TA = typename Config::TA;
    using TB = typename Config::TB;
    static constexpr int kStage  = Config::kStage;
    static constexpr int kTileM  = Config::kTileM;
    static constexpr int kTileN  = Config::kTileN;
    static constexpr int kTileK  = Config::kTileK;
    static constexpr int kTmaBytes = kTileM * kTileK * sizeof(TA)
                                   + kTileN * kTileK * sizeof(TB);

    if (prod_tid == 0) {
        // 在函数内部构建 TMA 视图 (寄存器隔离)
        auto mA = tma_a.get_tma_tensor(make_shape(M, K));
        auto mB = tma_b.get_tma_tensor(make_shape(N, K));

        auto gA = local_tile(mA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(by, _));
        auto gB = local_tile(mB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(bx, _));

        auto sA = make_tensor(make_smem_ptr(reinterpret_cast<TA*>(smem_A_ptr)),
                              typename Config::SmemLayoutA{});
        auto sB = make_tensor(make_smem_ptr(reinterpret_cast<TB*>(smem_B_ptr)),
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
            // 等待 Consumer 释放 stage (WAR 安全)
            mbar_wait(&mbar_empty[stage], phase_e);

            // 发起 TMA，设置 mbar_full 预期字节数
            mbar_arrive_and_expect_tx(&mbar_full[stage], kTmaBytes);
            copy(tma_a.with(mbar_full[stage]), tAgA(_, _, _, k), tAsA(_, _, _, stage));
            copy(tma_b.with(mbar_full[stage]), tBgB(_, _, _, k), tBsB(_, _, _, stage));

            stage = (stage + 1) % kStage;
            if (stage == 0) phase_e ^= 1;
        }
    }
}

// ============================================================================
// pp_fp8_consumer_loop  —  Consumer WarpGroup 主循环 + Epilogue
//
// __forceinline__: 避免 C7510 (wgmma pipeline 不跨函数边界)
//
// FP8 Epilogue: FP32 累加器直接 copy(tCrC, tCgC)
//   SM90 wgmma FP8: acc = FP32 register tensor
//   gC = FP32 GMEM tensor
//   copy(tCrC, tCgC) → vectorized 128-bit write to GMEM (no SMEM needed)
// ============================================================================
template <typename Config, typename SmemTensorA, typename SmemTensorB,
          typename GmemTensorC>
__device__ __forceinline__ void pp_fp8_consumer_loop(
    uint64_t* __restrict__ mbar_full,
    uint64_t* __restrict__ mbar_empty,
    SmemTensorA const& sA,
    SmemTensorB const& sB,
    GmemTensorC const& gC,
    int wg_tid,
    int num_k_tiles)
{
    static constexpr int kStage = Config::kStage;

    typename Config::TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(wg_tid);

    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    int stage   = 0;
    int phase_f = 0;

    for (int k = 0; k < num_k_tiles; ++k) {
        // 全 128 线程等待 TMA 完成 (acquire 语义)
        mbar_wait(&mbar_full[stage], phase_f);

        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        gemm(tiled_mma,
             tCrA(_, _, _, stage),
             tCrB(_, _, _, stage),
             tCrC);
        warpgroup_commit_batch();

        warpgroup_wait<1>();
        warpgroup_fence_operand(tCrC);

        // 通知 Producer: prev_stage SMEM 已用完 (WAR 安全)
        int prev_stage = (stage - 1 + kStage) % kStage;
        mbar_arrive_if(&mbar_empty[prev_stage], (k > 0) && (wg_tid == 0));

        stage = (stage + 1) % kStage;
        if (stage == 0) phase_f ^= 1;
    }

    // Drain
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    // 释放最后一个 stage
    int last_stage = (stage - 1 + kStage) % kStage;
    mbar_arrive_if(&mbar_empty[last_stage], (num_k_tiles > 0) && (wg_tid == 0));

    // Epilogue: FP32 accumulator → FP32 GMEM
    // 直接 R→G copy (vectorized, 无需 SMEM 中转)
    copy(tCrC, tCgC);
}

// ============================================================================
// gemm_kernel_fp8_pingpong  —  kernel 本体
//
// __launch_bounds__(160, 2): 提示 ptxas 目标 occupancy=2
//   max_regs = floor(65536/(2×160)) = 204，超过 163，所以不会强制 spill
// ============================================================================
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__global__ void __launch_bounds__(Config::kNumThreadsPP, 2)
gemm_kernel_fp8_pingpong(
    float*                         __restrict__ Cptr,
    CUTLASS_GRID_CONSTANT TmaCopyA const tma_a,
    CUTLASS_GRID_CONSTANT TmaCopyB const tma_b,
    int M, int N, int K)
{
    using TA = typename Config::TA;
    using TB = typename Config::TB;
    static constexpr int kTileM = Config::kTileM;
    static constexpr int kTileN = Config::kTileN;
    static constexpr int kStage = Config::kStage;

    // ------------------------------------------------------------------
    // 共享内存布局
    // ------------------------------------------------------------------
    extern __shared__ char smem_buf[];

    constexpr size_t smem_bytes_A = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(TA);
    constexpr size_t smem_bytes_B = cute::cosize(typename Config::SmemLayoutB{}) * sizeof(TB);
    constexpr size_t mbar_offset  = (smem_bytes_A + smem_bytes_B + 7) & ~7;

    char*     smem_A_ptr = smem_buf;
    char*     smem_B_ptr = smem_buf + smem_bytes_A;
    uint64_t* mbar_full  = reinterpret_cast<uint64_t*>(smem_buf + mbar_offset);
    uint64_t* mbar_empty = mbar_full + kStage;

    int tid                = threadIdx.x;
    const bool is_producer = (tid >= 128);
    const int  wg_tid      = tid;         // Consumer: 0-127
    const int  prod_tid    = tid - 128;   // Producer: 0-31

    // ------------------------------------------------------------------
    // mbarrier 初始化
    // ------------------------------------------------------------------
    if (tid == 0) {
        for (int s = 0; s < kStage; ++s) {
            mbar_init(&mbar_full[s],  1);
            mbar_init(&mbar_empty[s], 1);
        }
        mbar_fence_init();
    }
    __syncthreads();

    // Consumer (wg_tid==0) 预先 arrive mbar_empty，让 Producer 第一轮无需等待
    if (!is_producer && wg_tid == 0) {
        for (int s = 0; s < kStage; ++s) {
            mbar_arrive(&mbar_empty[s]);
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // 分发
    // ------------------------------------------------------------------
    if (is_producer) {
        pp_fp8_producer_loop<Config>(
            mbar_full, mbar_empty,
            tma_a, tma_b,
            smem_A_ptr, smem_B_ptr,
            prod_tid,
            blockIdx.x, blockIdx.y,
            M, N, K);
    } else {
        // Consumer 路径: 只有 gC/sA/sB + wgmma 累加器 (无 TMA 视图寄存器)
        auto sA = make_tensor(make_smem_ptr(reinterpret_cast<TA*>(smem_A_ptr)),
                              typename Config::SmemLayoutA{});
        auto sB = make_tensor(make_smem_ptr(reinterpret_cast<TB*>(smem_B_ptr)),
                              typename Config::SmemLayoutB{});
        auto C  = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(N, _1{}));
        auto gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                             make_coord(blockIdx.y, blockIdx.x));

        int num_k_tiles = K / Config::kTileK;
        pp_fp8_consumer_loop<Config>(
            mbar_full, mbar_empty,
            sA, sB, gC,
            wg_tid, num_k_tiles);
    }

    __syncthreads();
}

} // namespace gemm_fp8_sm90
