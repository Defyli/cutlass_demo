#pragma once

// ============================================================================
// Kernel 5: gemm_kernel_pingpong_cluster  (Cluster=1x2 + TMA A-Multicast)
//
// 设计理念:
//   传统 PingPong (cluster_size=1):
//     每个 CTA 独立加载完整的 A(128×64) + B(128×64) tile
//     每 K 步: 1 个 CTA → kTmaBytes = A_bytes + B_bytes 的 TMA
//
//   Cluster (cluster_size=2, N 方向):
//     2 个 CTA 组成 cluster, 在 N 方向相邻
//     A tile 对两个 CTA 完全相同 (同 M tile, 不同 N tile)
//     TMA Multicast 策略:
//       make_tma_copy(SM90_TMA_LOAD_MULTICAST, mA, smem_layout_A, cluster_size=2)
//         → TMA descriptor 的 box 大小减半 (每次加载 A 的一半)
//       CTA0: get_slice(0) → 加载 A 的上半部分 (行 0..63), multicast 到两个 CTA
//       CTA1: get_slice(1) → 加载 A 的下半部分 (行 64..127), multicast 到两个 CTA
//       结果: 两个 CTA 协同完成完整 A tile 的加载, 总 TMA count 不变但并发度翻倍
//
//   实际收益:
//     - A 的加载并发度 × 2 (两个 CTA 同时加载 A 的不同部分)
//     - B 的加载: 各自独立, 无变化
//     - 整体 occupancy 不变 (96KB SMEM, occupancy=2)
//
//   Cluster launch 方式:
//     __cluster_dims__(2, 1, 1)
//     grid.x = N/kTileN, grid.y = M/kTileM
//     注意: cudaLaunchKernelExArgs 或 __cluster_dims__ 显式指定
//
// mbarrier 设计 (关键!):
//   死锁根因: 两个 CTA 各自调用 mbar_arrive_and_expect_tx(shared::cta), 但
//   TMA multicast 完成时会通知双方的 mbar。如果 CTA1 的 TMA 完成时 CTA1.mbar 的
//   tx_count 还未设置（CTA1 未调用 expect_tx），则 tx_count 下溢 → 永远不归零 → 死锁。
//
//   修复: leader (cta_rank=0) 负责 arrive 所有 CTA 的 mbar_full:
//     - 本地 arrive: mbar_arrive_and_expect_tx(本 CTA mbar_full, kExpectTx)
//     - 远端 arrive: mbar_arrive_and_expect_tx_remote(远端 CTA mbar_full, kExpectTx, cta_id=1)
//   follower (cta_rank=1) 不调用 expect_tx, 直接发起 TMA。
//   这样保证两个 CTA 的 mbar 都在 TMA 发起前被设置好 tx_count。
//
// 线程布局: 与 PingPong v5 相同 (160 线程, Consumer 0-127, Producer 128-159)
// SMEM: 96KB (kStage=3), occupancy=2 blocks/SM
// ============================================================================

#include "detail/config.cuh"
#include "detail/mbarrier.cuh"
#include "detail/kernel_pingpong.cuh"  // pp_consumer_loop
#include "cutlass/device_kernel.h"

namespace gemm_sm90 {

using namespace cute;

// ============================================================================
// pp_cluster_producer  —  Cluster-aware Producer
//
// 与 v5 的区别:
//   - tma_a 是 SM90_TMA_LOAD_MULTICAST (每次只加载 A 的 1/cluster_size 部分)
//   - cta_rank 决定 get_slice(cta_rank) (选择 A 的哪一部分)
//   - mcast_mask_a = 0b11 (两个 CTA 都接收 A 的 multicast)
//   - B 的 TMA: 各 CTA 独立加载自己的 B tile (不变)
//
// mbarrier 关键设计:
//   TMA multicast 完成时会同时通知 cluster 内所有 CTA 的 mbar_full。
//   为了避免竞争条件（某 CTA 的 TMA 通知到达时，目标 CTA 的 mbar 尚未设置 expect_tx，
//   导致 tx_count 下溢 → 永远不归零 → 死锁），采用 leader 统一设置方案：
//
//   leader (cta_rank=0):
//     1. arrive 本 CTA 的 mbar_full (shared::cta, 本地)
//     2. arrive 所有 follower CTA 的 mbar_full (shared::cluster, 远端)
//        → 保证所有 mbar 的 tx_count 在 TMA 发起前已设置好
//     3. 发起本 CTA 负责的 A multicast TMA + B TMA
//   follower (cta_rank=1):
//     1. 不调用 expect_tx (leader 已远程 arrive 了)
//     2. 发起本 CTA 负责的 A multicast TMA + B TMA
//
//   expect_tx 值分析:
//     每个 CTA 的 mbar_full 接收到的 TMA 字节通知 =
//       CTA0.A multicast (到本 CTA 的 smem_A 前半) : kABytesTotal/2
//       CTA1.A multicast (到本 CTA 的 smem_A 后半) : kABytesTotal/2
//       本 CTA.B TMA (到本 CTA 的 smem_B)          : kBBytes
//     合计 = kABytesTotal + kBBytes (与普通 PingPong 相同)
// ============================================================================
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__device__ __noinline__ void pp_cluster_producer(
    uint64_t* __restrict__ mbar_full,
    uint64_t* __restrict__ mbar_empty,
    TmaCopyA const& tma_a,    // SM90_TMA_LOAD_MULTICAST
    TmaCopyB const& tma_b,    // SM90_TMA_LOAD
    char* __restrict__ smem_A_ptr,
    char* __restrict__ smem_B_ptr,
    int prod_tid,
    int bx, int by,
    int cta_rank,   // 0 或 1 (cluster 内 N 方向排名)
    int M, int N, int K)
{
    static constexpr int kStage  = Config::kStage;
    static constexpr int kTileM  = Config::kTileM;
    static constexpr int kTileN  = Config::kTileN;
    static constexpr int kTileK  = Config::kTileK;
    using T = typename Config::T;

    static constexpr int kABytesTotal = kTileM * kTileK * sizeof(T);
    static constexpr int kBBytes = kTileN * kTileK * sizeof(T);
    // 每个 CTA 的 mbar_full 期待接收的总 TMA 字节数
    //   = kABytesTotal (来自 A multicast, 两个 CTA 协作写满完整 A tile)
    //   + kBBytes (本 CTA 自己的 B TMA)
    static constexpr int kExpectTx = kABytesTotal + kBBytes;

    // multicast mask: cluster 内所有 CTA 都接收 A (bit 0 = CTA0, bit 1 = CTA1)
    static constexpr uint16_t mcast_mask_a = 0x3;

    if (prod_tid == 0) {
        auto mA = tma_a.get_tma_tensor(make_shape(M, K));
        auto mB = tma_b.get_tma_tensor(make_shape(N, K));

        // TMA A: cluster 内 cta_rank 决定加载 A 的哪一部分
        // tma_a 是 multicast 版本, get_slice(cta_rank) 选取 A 的 1/cluster_size 部分
        auto cta_tma_a = tma_a.get_slice(cta_rank);
        // TMA B: 普通, get_slice(0)
        auto cta_tma_b = tma_b.get_slice(Int<0>{});

        // A tile: by 相同 (M 方向), B tile: bx 不同 (N 方向)
        auto gA = local_tile(mA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(by, _));
        auto gB = local_tile(mB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(bx, _));

        auto sA = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_A_ptr)),
                              typename Config::SmemLayoutA{});
        auto sB = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_B_ptr)),
                              typename Config::SmemLayoutB{});

        // 注意: get_slice(cta_rank) 已经为 multicast 设置了正确的 smem 目标地址
        Tensor tAgA = cta_tma_a.partition_S(gA);
        Tensor tAsA = cta_tma_a.partition_D(as_position_independent_swizzle_tensor(sA));
        Tensor tBgB = cta_tma_b.partition_S(gB);
        Tensor tBsB = cta_tma_b.partition_D(as_position_independent_swizzle_tensor(sB));

        int num_k_tiles = K / kTileK;
        int stage   = 0;
        int phase_e = 0;

        for (int k = 0; k < num_k_tiles; ++k) {
            mbar_wait(&mbar_empty[stage], phase_e);

            // ---------------------------------------------------------------
            // expect_tx 设置 (关键: leader 负责所有 CTA 的 mbar_full arrive)
            //
            // 只有 cta_rank=0 (leader) 负责 arrive mbar_full:
            //   - 本 CTA 的 mbar_full: 使用 shared::cta (本地 arrive)
            //   - 其余 CTA 的 mbar_full: 使用 shared::cluster (远端 arrive)
            //
            // follower (cta_rank=1) 不调用 expect_tx。
            // 这避免了竞争: TMA 通知到达时 mbar 一定已设置 tx_count。
            // ---------------------------------------------------------------
            if (cta_rank == 0) {
                // arrive 本 CTA (CTA0) 的 mbar_full
                mbar_arrive_and_expect_tx(&mbar_full[stage], kExpectTx);
                // arrive 远端 CTA (CTA1) 的 mbar_full (cluster 内 ID=1)
                mbar_arrive_and_expect_tx_remote(&mbar_full[stage], kExpectTx, /*cta_id=*/1);
            }

            // A: TMA Multicast (写入 cluster 内所有 CTA 的 smem_A 对应部分)
            // 每个 CTA 各自发起一半 A tile 的加载, multicast 让双方 smem_A 都写满
            copy(tma_a.with(mbar_full[stage], mcast_mask_a),
                 tAgA(_, _, _, k), tAsA(_, _, _, stage));
            // B: 普通 TMA (各 CTA 独立加载自己的 B tile)
            copy(tma_b.with(mbar_full[stage]),
                 tBgB(_, _, _, k), tBsB(_, _, _, stage));

            stage = (stage + 1) % kStage;
            if (stage == 0) phase_e ^= 1;
        }
    }
}

// ============================================================================
// gemm_kernel_pingpong_cluster  —  Cluster=1x2 kernel
//
// Grid 设置:
//   __cluster_dims__(1, 2, 1): Y 方向 cluster_size=2
//   grid.y = M/kTileM × cluster_size_y = M/kTileM × 2 个实际 block
//   但实际上 grid.y 按 cluster 数量算:
//     grid.y 实际 = M/kTileM  (每个 cluster 覆盖 1 个 M tile 和 2 个 N tile)
//
// 等等, 这里我换一个更清晰的方案:
//   __cluster_dims__(2, 1, 1): X 方向 cluster_size=2 (N 方向)
//   grid.x = N/kTileN (实际 block 数量, cluster_size=2 所以 grid.x 必须是偶数)
//   grid.y = M/kTileM
//   cluster 编号 = blockIdx.x / 2
//   cta_rank = blockIdx.x % 2 (在 cluster 内的 N 方向排名)
//   bx = blockIdx.x (全局 N tile 索引)
//   by = blockIdx.y (全局 M tile 索引)
//
// 每个 CTA:
//   cta_rank = blockIdx.x % 2
//   bx = blockIdx.x (N tile 索引)
//   by = blockIdx.y (M tile 索引)
//
// 注意: 以上 grid 设置与普通 PingPong 相同 (N/kTileN × M/kTileM)
//   区别只在于内部的 TMA 调用 (multicast for A)
// ============================================================================
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__global__ void __launch_bounds__(Config::kNumThreadsPP, 2)
__cluster_dims__(2, 1, 1)
gemm_kernel_pingpong_cluster(
    float*       __restrict__ Cptr,
    CUTLASS_GRID_CONSTANT TmaCopyA const tma_a,  // SM90_TMA_LOAD_MULTICAST
    CUTLASS_GRID_CONSTANT TmaCopyB const tma_b,  // SM90_TMA_LOAD
    int M, int N, int K)
{
    using T = typename Config::T;
    static constexpr int kTileM = Config::kTileM;
    static constexpr int kTileN = Config::kTileN;
    static constexpr int kStage = Config::kStage;

    extern __shared__ char smem_buf[];

    constexpr size_t smem_bytes_A = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(T);
    constexpr size_t smem_bytes_B = cute::cosize(typename Config::SmemLayoutB{}) * sizeof(T);
    constexpr size_t mbar_offset  = (smem_bytes_A + smem_bytes_B + 7) & ~size_t(7);

    char*     smem_A_ptr = smem_buf;
    char*     smem_B_ptr = smem_buf + smem_bytes_A;
    uint64_t* mbar_full  = reinterpret_cast<uint64_t*>(smem_buf + mbar_offset);
    uint64_t* mbar_empty = mbar_full + kStage;

    int tid               = threadIdx.x;
    const bool is_producer = (tid >= 128);
    const int  wg_tid      = tid;
    const int  prod_tid    = tid - 128;

    // cluster 内 CTA rank: x 方向 (N 方向)
    // cta_rank = blockIdx.x % 2 (因为 __cluster_dims__(2,1,1))
    int cta_rank = blockIdx.x % 2;
    int bx = blockIdx.x;   // N 方向 tile 索引 (全局)
    int by = blockIdx.y;   // M 方向 tile 索引

    // ------------------------------------------------------------------
    // mbarrier 初始化
    // arrive_count=1: 每个 CTA 自己发起 expect_tx arrive
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
    // 分发
    // ------------------------------------------------------------------
    if (is_producer) {
        pp_cluster_producer<Config>(
            mbar_full, mbar_empty,
            tma_a, tma_b,
            smem_A_ptr, smem_B_ptr,
            prod_tid,
            bx, by,
            cta_rank,
            M, N, K);
    } else {
        auto sA = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_A_ptr)),
                              typename Config::SmemLayoutA{});
        auto sB = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_B_ptr)),
                              typename Config::SmemLayoutB{});
        auto C  = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(N, _1{}));
        auto gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                             make_coord(by, bx));

        int num_k_tiles = K / Config::kTileK;
        pp_consumer_loop<Config>(
            mbar_full, mbar_empty,
            sA, sB, gC,
            wg_tid, num_k_tiles);
    }

    __syncthreads();
}

// SMEM: 与普通 PingPong 相同
template <typename Config>
constexpr size_t get_smem_size_pingpong_cluster() {
    return get_smem_size_pingpong<Config>();
}

} // namespace gemm_sm90
