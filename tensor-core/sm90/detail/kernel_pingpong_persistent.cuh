#pragma once

// ============================================================================
// Kernel 4: gemm_kernel_pingpong_persistent  (Persistent Kernel + TMA + GMMA)
//
// 动机:
//   当前 PingPong v5: Grid = 32×32 = 1024 blocks, H20-3e 有 78 SM,
//   occupancy=2 → 156 blocks 同时活跃, 需要 ≈6.6 waves 才能完成所有 tiles.
//   每次 wave 切换有调度开销, 且最后一波可能 load imbalance.
//
//   Persistent Kernel: 只发射 78×2=156 blocks, 通过全局原子计数器
//   动态获取 tiles, 消除 wave 切换, 实现完美负载均衡.
//
// 设计:
//   - 外层 while 循环: 每次 atomicAdd(&tile_counter, 1) 获取 tile_id
//   - 每个 tile 内部: 重新初始化 mbarrier, 执行 Producer/Consumer 协作
//   - mbarrier 复用同一块 SMEM (每 tile 开始前 mbar_init 复位)
//   - 寄存器保持与 v5 相同: Producer __noinline__, Consumer __forceinline__
//
// 关键权衡:
//   + 消除 wave 切换开销 (~几微秒/wave × 6.6 waves)
//   + 负载均衡更好 (work stealing)
//   - 每 tile 多一次 syncthreads + mbar_init
//   - 每 tile 多一次 atomicAdd (全局内存 ~100ns)
//
// SMEM: 96KB (kStage=3), occupancy=2 blocks/SM
// ============================================================================

#include "detail/config.cuh"
#include "detail/mbarrier.cuh"
#include "cutlass/device_kernel.h"   // CUTLASS_GRID_CONSTANT

namespace gemm_sm90 {

using namespace cute;

// ============================================================================
// pp_pers_producer  —  处理一个 tile 的 K-loop Producer
// __noinline__ 保持寄存器隔离 (与 v5 相同)
// ============================================================================
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__device__ __noinline__ void pp_pers_producer(
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
    static constexpr int kStage    = Config::kStage;
    static constexpr int kTileM    = Config::kTileM;
    static constexpr int kTileN    = Config::kTileN;
    static constexpr int kTileK    = Config::kTileK;
    using T = typename Config::T;
    static constexpr int kTmaBytes = kTileM * kTileK * sizeof(T)
                                   + kTileN * kTileK * sizeof(T);

    if (prod_tid == 0) {
        auto mA = tma_a.get_tma_tensor(make_shape(M, K));
        auto mB = tma_b.get_tma_tensor(make_shape(N, K));

        auto gA = local_tile(mA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(by, _));
        auto gB = local_tile(mB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(bx, _));

        using T_ = typename Config::T;
        auto sA = make_tensor(make_smem_ptr(reinterpret_cast<T_*>(smem_A_ptr)),
                              typename Config::SmemLayoutA{});
        auto sB = make_tensor(make_smem_ptr(reinterpret_cast<T_*>(smem_B_ptr)),
                              typename Config::SmemLayoutB{});

        auto cta_tma_a = tma_a.get_slice(Int<0>{});
        auto cta_tma_b = tma_b.get_slice(Int<0>{});

        Tensor tAgA = cta_tma_a.partition_S(gA);
        Tensor tAsA = cta_tma_a.partition_D(as_position_independent_swizzle_tensor(sA));
        Tensor tBgB = cta_tma_b.partition_S(gB);
        Tensor tBsB = cta_tma_b.partition_D(as_position_independent_swizzle_tensor(sB));

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
// gemm_kernel_pingpong_persistent  —  Persistent Kernel 本体
//
// 调用:
//   dim3 grid(num_sm * 2);          // 156 blocks, 填满所有 SM
//   dim3 block(Config::kNumThreadsPP);  // 160 线程
//   gemm_kernel_pingpong_persistent<<<grid, block, smem_size>>>(
//       Cptr, d_tile_counter, tma_a, tma_b, M, N, K, tiles_n, tiles_m);
//
//   注意: d_tile_counter 需要在每次 kernel launch 前 cudaMemset 为 0
//
// 寄存器:
//   Consumer 线程路径下包含 wgmma 累加器 (tCrC) + 少量循环变量
//   Producer 路径下 __noinline__ 隔离 TMA 视图寄存器
//   预期: 154-160 reg/thread (与 v5 相同)
// ============================================================================
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__global__ void __launch_bounds__(Config::kNumThreadsPP, 2)
gemm_kernel_pingpong_persistent(
    float*       __restrict__ Cptr,
    int*         __restrict__ tile_counter,  // 全局原子计数器, 初始=0
    CUTLASS_GRID_CONSTANT TmaCopyA const tma_a,
    CUTLASS_GRID_CONSTANT TmaCopyB const tma_b,
    int M, int N, int K,
    int num_tiles_n,   // N / kTileN
    int num_tiles_m)   // M / kTileM
{
    using T = typename Config::T;
    static constexpr int kTileM = Config::kTileM;
    static constexpr int kTileN = Config::kTileN;
    static constexpr int kStage = Config::kStage;

    // ------------------------------------------------------------------
    // SMEM 布局 (与 v5 相同)
    // ------------------------------------------------------------------
    extern __shared__ char smem_buf[];

    constexpr size_t smem_bytes_A = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(T);
    constexpr size_t smem_bytes_B = cute::cosize(typename Config::SmemLayoutB{}) * sizeof(T);
    // mbarrier 需要 8B 对齐; 在 A+B 之后对齐
    constexpr size_t mbar_offset  = (smem_bytes_A + smem_bytes_B + 7) & ~size_t(7);

    char*     smem_A_ptr = smem_buf;
    char*     smem_B_ptr = smem_buf + smem_bytes_A;
    uint64_t* mbar_full  = reinterpret_cast<uint64_t*>(smem_buf + mbar_offset);
    uint64_t* mbar_empty = mbar_full + kStage;

    int tid            = threadIdx.x;
    const bool is_producer = (tid >= 128);
    const int  wg_tid      = tid;
    const int  prod_tid    = tid - 128;

    int total_tiles = num_tiles_m * num_tiles_n;

    // 每 tile 复用同一块 SMEM 的 sA/sB tensor (layout 不变)
    auto sA = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_A_ptr)),
                          typename Config::SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_B_ptr)),
                          typename Config::SmemLayoutB{});

    // Consumer 需要的 MMA 分区 (只做一次)
    typename Config::TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(wg_tid);
    Tensor tCsA  = thr_mma.partition_A(sA);
    Tensor tCsB  = thr_mma.partition_B(sB);
    Tensor tCrA  = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB  = thr_mma.make_fragment_B(tCsB);

    // ------------------------------------------------------------------
    // 外层 Persistent 循环
    // tile_id 广播: 利用动态 SMEM 末尾 (mbarrier 之后) 的 4 字节存储
    // 确保对齐: mbar_offset 对齐到 8B, mbar_bytes = 2*kStage*8B, 末尾再加 4B
    // 注意: mbar_full/empty 初始化在每次 tile 开始时做, 所以末尾空间无冲突
    // ------------------------------------------------------------------
    constexpr size_t tile_id_offset = mbar_offset + 2 * kStage * sizeof(uint64_t);
    int* smem_tile_id = reinterpret_cast<int*>(smem_buf + tile_id_offset);

    while (true) {
        // ---- 获取下一个 tile (tid==0 原子自增, SMEM 广播给全 block) ----
        if (tid == 0) {
            *smem_tile_id = atomicAdd(tile_counter, 1);
        }
        __syncthreads();

        int tile_id = *smem_tile_id;
        if (tile_id >= total_tiles) break;

        int bx = tile_id % num_tiles_n;
        int by = tile_id / num_tiles_n;

        // ---- 重新初始化 mbarrier ----
        if (tid == 0) {
            for (int s = 0; s < kStage; ++s) {
                mbar_init(&mbar_full[s],  1);
                mbar_init(&mbar_empty[s], 1);
            }
            mbar_fence_init();
        }
        __syncthreads();

        // Consumer (wg_tid==0) 预先 arrive empty (让 Producer 无需等待前 kStage 轮)
        if (!is_producer && wg_tid == 0) {
            for (int s = 0; s < kStage; ++s) {
                mbar_arrive(&mbar_empty[s]);
            }
        }
        __syncthreads();

        // ---- 执行当前 tile ----
        if (is_producer) {
            pp_pers_producer<Config>(
                mbar_full, mbar_empty,
                tma_a, tma_b,
                smem_A_ptr, smem_B_ptr,
                prod_tid,
                bx, by,
                M, N, K);
        } else {
            // Consumer: 构建 gC (每次 tile 不同)
            auto C  = make_tensor(make_gmem_ptr(Cptr),
                                  make_shape(M, N), make_stride(N, _1{}));
            auto gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                                 make_coord(by, bx));
            Tensor tCgC = thr_mma.partition_C(gC);

            // 累加器清零 (每 tile 重新开始)
            Tensor tCrC = thr_mma.make_fragment_C(tCgC);
            clear(tCrC);

            // K 方向循环
            int num_k_tiles = K / Config::kTileK;
            int stage   = 0;
            int phase_f = 0;

            for (int k = 0; k < num_k_tiles; ++k) {
                mbar_wait(&mbar_full[stage], phase_f);

                warpgroup_fence_operand(tCrC);
                warpgroup_arrive();
                gemm(tiled_mma, tCrA(_, _, _, stage), tCrB(_, _, _, stage), tCrC);
                warpgroup_commit_batch();

                warpgroup_wait<1>();
                warpgroup_fence_operand(tCrC);

                int prev_stage = (stage - 1 + kStage) % kStage;
                mbar_arrive_if(&mbar_empty[prev_stage], (k > 0) && (wg_tid == 0));

                stage = (stage + 1) % kStage;
                if (stage == 0) phase_f ^= 1;
            }

            // Drain
            warpgroup_wait<0>();
            warpgroup_fence_operand(tCrC);

            int last_stage = (stage - 1 + kStage) % kStage;
            mbar_arrive_if(&mbar_empty[last_stage], (num_k_tiles > 0) && (wg_tid == 0));

            // Epilogue: 写回
            copy(tCrC, tCgC);
        }

        __syncthreads();
    }  // end while (tile_id < total_tiles)
}

// ============================================================================
// get_smem_size_pingpong_persistent: 与普通 PingPong 相同 SMEM 大小
// (但注意有额外的 __shared__ int smem_tile_id, 编译器自动处理)
// ============================================================================
// get_smem_size_pingpong_persistent: 在普通 PingPong SMEM 基础上额外加 4B (tile_id)
template <typename Config>
constexpr size_t get_smem_size_pingpong_persistent() {
    using T = typename Config::T;
    constexpr size_t smem_bytes_A = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(T);
    constexpr size_t smem_bytes_B = cute::cosize(typename Config::SmemLayoutB{}) * sizeof(T);
    constexpr size_t mbar_offset  = (smem_bytes_A + smem_bytes_B + 7) & ~size_t(7);
    constexpr size_t mbar_bytes   = 2 * Config::kStage * sizeof(uint64_t);
    // 末尾追加 4B 存储 tile_id (供广播使用)
    return mbar_offset + mbar_bytes + sizeof(int);
}

} // namespace gemm_sm90
