#pragma once

// ============================================================================
// FP8 Persistent Ping-Pong Kernel (v7)
//
// 设计目标: 消除 Wave Quantization 损失
//
// 关键 insight: 对于 H20-3e (78 SMs, occupancy=2):
//   8192×8192: total_tiles=4096, persistent_grid=156
//   ceil(4096/156)=27 waves → 最后一波只有 40/156=26% 的 SM 利用率
//   wave quantization 损失 ≈ 74%×(1/27) ≈ 2.7%
//
// v7 实现: Persistent Kernel（每次 tile 切换时最小化同步开销）
//   - Grid size = num_SMs × 2（固定占满所有 SM）
//   - 每个 block 持续 atomicAdd 领取 tile 任务
//   - 每次 tile 切换只需：
//     1. __syncthreads() × 1（atomic + broadcast）
//     2. mbar_init × 2×kStage（轻量级，无需 fence_init）
//        → 用 CTA-scope init 替代 cluster-scope fence
//     3. Consumer 预 arrive mbar_empty × kStage
//     4. __syncthreads() × 1
//
// 寄存器策略: 与 v5 完全相同
//   Consumer __forceinline__, Producer __noinline__
//   __launch_bounds__(160, 2) → 154 regs, occupancy=2
// ============================================================================

#include "config_fp8.cuh"
#include "mbarrier.cuh"
#include "cutlass/device_kernel.h"   // CUTLASS_GRID_CONSTANT

namespace gemm_fp8_sm90 {

using namespace cute;

// 轻量级 mbarrier init（CTA scope，无 cluster fence，适用于单 CTA kernel）
CUTE_DEVICE void mbar_init_cta(uint64_t* mbar, int arrive_count) {
    uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(smem_ptr), "r"(arrive_count));
}

// ============================================================================
// pp_fp8_persistent_producer  —  Producer 单个 tile 的 TMA 流水线
// __noinline__: 隔离 TMA 视图寄存器
// ============================================================================
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__device__ __noinline__ void pp_fp8_persistent_producer(
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
// pp_fp8_persistent_consumer  —  Consumer 单个 tile 的 wgmma 流水线 + epilogue
// __forceinline__: 避免 C7510
// ============================================================================
template <typename Config, typename SmemTensorA, typename SmemTensorB,
          typename GmemTensorC>
__device__ __forceinline__ void pp_fp8_persistent_consumer(
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

        int prev_stage = (stage - 1 + kStage) % kStage;
        mbar_arrive_if(&mbar_empty[prev_stage], (k > 0) && (wg_tid == 0));

        stage = (stage + 1) % kStage;
        if (stage == 0) phase_f ^= 1;
    }

    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    int last_stage = (stage - 1 + kStage) % kStage;
    mbar_arrive_if(&mbar_empty[last_stage], (num_k_tiles > 0) && (wg_tid == 0));

    copy(tCrC, tCgC);
}

// ============================================================================
// gemm_kernel_fp8_persistent  —  Persistent Ping-Pong kernel (v7)
//
// tile loop 中每次切换开销：
//   1. __syncthreads() × 1（tile_id broadcast）
//   2. mbar_init × 2×kStage（CTA scope，轻量）
//   3. Consumer 预 arrive mbar_empty × kStage
//   4. __syncthreads() × 1
//
// 无 mbar_fence_init（cluster fence），替代为 CTA scope init
// ============================================================================
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__global__ void __launch_bounds__(Config::kNumThreadsPP, 2)
gemm_kernel_fp8_persistent(
    float*                         __restrict__ Cptr,
    CUTLASS_GRID_CONSTANT TmaCopyA const tma_a,
    CUTLASS_GRID_CONSTANT TmaCopyB const tma_b,
    int* __restrict__ tile_counter,
    int total_tiles,
    int num_n_tiles,
    int M, int N, int K)
{
    using TA = typename Config::TA;
    using TB = typename Config::TB;
    static constexpr int kTileM = Config::kTileM;
    static constexpr int kTileN = Config::kTileN;
    static constexpr int kStage = Config::kStage;

    // ------------------------------------------------------------------
    // 共享内存布局: 与 v5 相同 + 4 bytes for tile_id
    // ------------------------------------------------------------------
    extern __shared__ char smem_buf[];

    constexpr size_t smem_bytes_A = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(TA);
    constexpr size_t smem_bytes_B = cute::cosize(typename Config::SmemLayoutB{}) * sizeof(TB);
    constexpr size_t mbar_offset  = (smem_bytes_A + smem_bytes_B + 7) & ~7;

    char*     smem_A_ptr = smem_buf;
    char*     smem_B_ptr = smem_buf + smem_bytes_A;
    uint64_t* mbar_full  = reinterpret_cast<uint64_t*>(smem_buf + mbar_offset);
    uint64_t* mbar_empty = mbar_full + kStage;
    // tile_id 存在 mbar 之后（4 bytes，对齐到 4 bytes 即可）
    int* tile_id_ptr     = reinterpret_cast<int*>(mbar_empty + kStage);

    int tid                = threadIdx.x;
    const bool is_producer = (tid >= 128);
    const int  wg_tid      = tid;         // Consumer: 0-127
    const int  prod_tid    = tid - 128;   // Producer: 0-31

    // ------------------------------------------------------------------
    // 外层 tile loop
    // ------------------------------------------------------------------
    while (true) {
        // ----------------------------------------------------------------
        // 步骤 1: 领取 tile_id（tid==0 做 atomic，__syncthreads 广播）
        // ----------------------------------------------------------------
        if (tid == 0) {
            *tile_id_ptr = atomicAdd(tile_counter, 1);
        }
        __syncthreads();

        int tile_id = *tile_id_ptr;
        if (tile_id >= total_tiles) break;

        int bx = tile_id % num_n_tiles;
        int by = tile_id / num_n_tiles;

        // ----------------------------------------------------------------
        // 步骤 2: 重新初始化 mbarrier（与 v5 初始化逻辑完全一致）
        // 用 CTA-scope init（不需要 cluster fence，单 CTA kernel 安全）
        // ----------------------------------------------------------------
        if (tid == 0) {
            for (int s = 0; s < kStage; ++s) {
                mbar_init_cta(&mbar_full[s],  1);
                mbar_init_cta(&mbar_empty[s], 1);
            }
            // fence.mbarrier_init.release.cluster 确保 mbar init 对同 cluster 可见
            // 单 CTA 无 cluster 场景下等价于 CTA-scope fence
            asm volatile("fence.mbarrier_init.release.cluster;\n" :::);
        }
        __syncthreads();

        // ----------------------------------------------------------------
        // 步骤 3: Consumer 预 arrive mbar_empty（与 v5 完全一致）
        // ----------------------------------------------------------------
        if (!is_producer && wg_tid == 0) {
            for (int s = 0; s < kStage; ++s) {
                mbar_arrive(&mbar_empty[s]);
            }
        }
        __syncthreads();

        // ----------------------------------------------------------------
        // 步骤 4: Consumer / Producer 各自处理当前 tile
        // ----------------------------------------------------------------
        if (is_producer) {
            pp_fp8_persistent_producer<Config>(
                mbar_full, mbar_empty,
                tma_a, tma_b,
                smem_A_ptr, smem_B_ptr,
                prod_tid,
                bx, by,
                M, N, K);
        } else {
            auto sA = make_tensor(make_smem_ptr(reinterpret_cast<TA*>(smem_A_ptr)),
                                  typename Config::SmemLayoutA{});
            auto sB = make_tensor(make_smem_ptr(reinterpret_cast<TB*>(smem_B_ptr)),
                                  typename Config::SmemLayoutB{});
            auto C  = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(N, _1{}));
            auto gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                                 make_coord(by, bx));

            int num_k_tiles = K / Config::kTileK;
            pp_fp8_persistent_consumer<Config>(
                mbar_full, mbar_empty,
                sA, sB, gC,
                wg_tid, num_k_tiles);
        }

        __syncthreads();
    }
}

} // namespace gemm_fp8_sm90
