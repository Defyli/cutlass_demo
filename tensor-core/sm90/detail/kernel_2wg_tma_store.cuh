#pragma once

// ============================================================================
// kernel_2wg_tma_store.cuh
//
// 包含两个基于 hpc-ops 经验的优化 kernel:
//
//   版本 A: gemm_kernel_pingpong_v6 (160 线程)
//     基于 v5 Ping-Pong, 升级 Epilogue:
//       旧: copy(tCrC, tCgC)  → 逐元素 STG 写 FP32 到 GMEM
//       新: FP32 SMEM 暂存 + TMA Store (S→G, FP32)
//     - 输出类型: FP32
//     - SMEM C buffer: float, row-major layout (kTileM×kTileN×4B = 64KB)
//       注意: kStage=3 时 A(48) + B(48) + C(64) = 160KB > 128KB
//             因此 v6 改为 kStage=2: A(32) + B(32) + C(64) + mbar ≈ 128KB ✓
//     - TMA Store 是异步的, 不阻塞 Math WG
//
//   版本 B: gemm_kernel_2wg_pingpong (384 线程)
//     真正的双 WG Cooperative GEMM:
//     - 每个 block 同时处理 2 个相邻 N tile (bx*2 和 bx*2+1)
//     - grid.x = N/(kTileN*2) (减半)
//     - 2 Math WG (WG0 → tile0, WG1 → tile1) + 1 Load WG
//     - Load WG 一次 TMA 加载 A+B0+B1 (共享 A, 分别加载 B)
//     - warpgroup_reg_alloc<168> (Math WG) + warpgroup_reg_dealloc<24> (Load WG)
//     - 两个 WG 并行执行 wgmma, 充分利用 SM 的计算资源
//     - 输出类型: FP32 (直接 STG 写回, epilogue 简单)
//     - 总 SMEM (kStage=2): A(32) + B0(32) + B1(32) + mbar ≈ 96KB ✓
//
// ============================================================================

#include "detail/config.cuh"
#include "detail/mbarrier.cuh"
#include "cutlass/device_kernel.h"
#include "cutlass/arch/reg_reconfig.h"

namespace gemm_sm90 {

using namespace cute;

// ============================================================================
// 辅助函数
// ============================================================================

// TMA Store fence: 保证 SMEM 写入对 TMA 硬件可见
// fence.proxy.async.shared::cta 使 TMA 硬件看到 SMEM 写入
__device__ __forceinline__ void tma_store_fence() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// TMA Store wait: 等待所有 in-flight TMA Store 完成
// cute::tma_store_wait<0>() = 等待直到 in-flight store 数量 <= 0 (全部完成)
__device__ __forceinline__ void tma_store_wait_all() {
    cute::tma_store_wait<0>();
}

// ============================================================================
// ─────────────────────────────────────────────────────────────────────────────
// 版本 A: Ping-Pong + TMA Store Epilogue (160 线程)
// ─────────────────────────────────────────────────────────────────────────────
// ============================================================================

// ----------------------------------------------------------------------------
// pp_v6_producer — 与 v5 pp_producer_loop 完全相同 (__noinline__ 隔离寄存器)
// ----------------------------------------------------------------------------
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__device__ __noinline__ void pp_v6_producer(
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

        auto sA = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_A_ptr)),
                              typename Config::SmemLayoutA{});
        auto sB = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_B_ptr)),
                              typename Config::SmemLayoutB{});

        Tensor tAgA = tma_a.get_slice(Int<0>{}).partition_S(gA);
        Tensor tAsA = tma_a.get_slice(Int<0>{}).partition_D(
            as_position_independent_swizzle_tensor(sA));
        Tensor tBgB = tma_b.get_slice(Int<0>{}).partition_S(gB);
        Tensor tBsB = tma_b.get_slice(Int<0>{}).partition_D(
            as_position_independent_swizzle_tensor(sB));

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

// ----------------------------------------------------------------------------
// pp_v6_consumer — 纯计算 Consumer WG (__forceinline__)
//
// 只做 wgmma 主循环，不包含 epilogue。
// 完成后累加器保留在 tCrC 寄存器中，由调用方负责写回。
//
// 设计原因:
//   原来 epilogue 使用 bar.sync/barrier.cta.sync 命名屏障同步 128 个 Consumer 线程，
//   但这些指令在本服务器的驱动版本上触发 Illegal instruction。
//   改为在主 kernel 中用 __syncthreads() 实现全 CTA 同步，完全规避命名屏障。
// ----------------------------------------------------------------------------
template <typename Config, typename SmemTensorA, typename SmemTensorB,
          typename SmemTensorC>
__device__ __forceinline__ void pp_v6_consumer(
    uint64_t* __restrict__ mbar_full,
    uint64_t* __restrict__ mbar_empty,
    SmemTensorA const& sA,
    SmemTensorB const& sB,
    SmemTensorC& sC,   // float SMEM C buffer (用于 epilogue r2s, 调用方负责 TMA Store)
    int wg_tid,
    int num_k_tiles)
{
    static constexpr int kStage = Config::kStage;

    typename Config::TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(wg_tid);

    // A/B partition (从 SMEM 读)
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    // 累加器 (FP32): 用 sC 推导形状，但计算时只写寄存器
    Tensor tCsC = thr_mma.partition_C(sC);
    Tensor tCrC = thr_mma.make_fragment_C(tCsC);
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

    // Drain: 等最后一条 wgmma 完成
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    int last_stage = (stage - 1 + kStage) % kStage;
    mbar_arrive_if(&mbar_empty[last_stage], (num_k_tiles > 0) && (wg_tid == 0));

    // -------------------------------------------------------------------------
    // Epilogue: 累加器寄存器 → float SMEM
    // 注意: 此时不做任何 __syncthreads/bar.sync，由调用方 (主 kernel) 负责同步
    // -------------------------------------------------------------------------
    {
        auto tiled_r2s = make_tiled_copy_C(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, float>{},
            tiled_mma
        );
        auto thread_r2s = tiled_r2s.get_slice(wg_tid);
        Tensor tRS_rC = thread_r2s.retile_S(tCrC);
        Tensor tRS_sC = thread_r2s.partition_D(sC);
        copy(tiled_r2s, tRS_rC, tRS_sC);
    }
    // 返回后 sC 中已有 FP32 结果，主 kernel 用 __syncthreads() 同步后发起 TMA Store
}

// ----------------------------------------------------------------------------
// gemm_kernel_pingpong_tma_store — 版本 A 主 kernel (160 线程)
//
// 输出: FP32 (float 累加器 → float SMEM → TMA Store → FP32 GMEM)
// SMEM: A(32KB, kStage=2) + B(32KB) + mbar(32B) + C(64KB) ≈ 128KB ✓
//   注意: 版本A使用 PPTmaStoreConfig (kStage=2) 来控制 SMEM 总量
// ----------------------------------------------------------------------------
template <typename Config, typename TmaCopyA, typename TmaCopyB, typename TmaStoreC>
__global__ void __launch_bounds__(Config::kNumThreadsPP, 2)
gemm_kernel_pingpong_tma_store(
    CUTLASS_GRID_CONSTANT TmaStoreC const tma_c,
    CUTLASS_GRID_CONSTANT TmaCopyA  const tma_a,
    CUTLASS_GRID_CONSTANT TmaCopyB  const tma_b,
    int M, int N, int K)
{
    using T = typename Config::T;
    static constexpr int kTileM = Config::kTileM;
    static constexpr int kTileN = Config::kTileN;
    static constexpr int kStage = Config::kStage;

    extern __shared__ char smem_buf[];

    // SMEM 布局计算
    constexpr size_t smem_bytes_A  = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(T);
    constexpr size_t smem_bytes_B  = cute::cosize(typename Config::SmemLayoutB{}) * sizeof(T);
    constexpr size_t mbar_offset   = (smem_bytes_A + smem_bytes_B + 7) & ~7UL;
    constexpr size_t mbar_bytes    = 2 * kStage * sizeof(uint64_t);
    // C buffer: 128B 对齐 (TMA 要求), float row-major layout
    constexpr size_t smem_C_offset = (mbar_offset + mbar_bytes + 127) & ~127UL;

    char*     smem_A_ptr = smem_buf;
    char*     smem_B_ptr = smem_buf + smem_bytes_A;
    uint64_t* mbar_full  = reinterpret_cast<uint64_t*>(smem_buf + mbar_offset);
    uint64_t* mbar_empty = mbar_full + kStage;
    float*    smem_C_ptr = reinterpret_cast<float*>(smem_buf + smem_C_offset);

    int tid = threadIdx.x;
    const bool is_producer = (tid >= 128);
    const int  wg_tid      = tid;
    const int  prod_tid    = tid - 128;

    // mbarrier 初始化
    if (tid == 0) {
        for (int s = 0; s < kStage; ++s) {
            mbar_init(&mbar_full[s],  1);
            mbar_init(&mbar_empty[s], 1);
        }
        mbar_fence_init();
    }
    __syncthreads();

    if (!is_producer && wg_tid == 0) {
        for (int s = 0; s < kStage; ++s) mbar_arrive(&mbar_empty[s]);
    }
    __syncthreads();

    // SMEM 张量构建
    auto sA = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_A_ptr)),
                          typename Config::SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_B_ptr)),
                          typename Config::SmemLayoutB{});
    // C buffer: float row-major layout (kTileM, kTileN), stride (kTileN, 1)
    auto sC = make_tensor(make_smem_ptr(smem_C_ptr),
                          make_layout(make_shape(Int<kTileM>{}, Int<kTileN>{}),
                                      make_stride(Int<kTileN>{}, _1{})));

    int num_k_tiles = K / Config::kTileK;

    if (is_producer) {
        pp_v6_producer<Config>(
            mbar_full, mbar_empty,
            tma_a, tma_b,
            smem_A_ptr, smem_B_ptr,
            prod_tid,
            blockIdx.x, blockIdx.y,
            M, N, K);
    } else {
        // Consumer: 纯计算，Epilogue 仅做 R2S (累加器寄存器 → float SMEM)
        // TMA Store 由主 kernel 在 __syncthreads() 之后统一发起
        pp_v6_consumer<Config>(
            mbar_full, mbar_empty,
            sA, sB, sC,
            wg_tid,
            num_k_tiles);
    }

    // -------------------------------------------------------------------------
    // __syncthreads(): 保证所有 Consumer 线程完成 R2S 写入 sC
    // (Producer WG 线程此时已完成 TMA Load 循环, 不写 sC, 参与同步即可)
    // -------------------------------------------------------------------------
    __syncthreads();

    // -------------------------------------------------------------------------
    // TMA Store: tid 0 将 float SMEM → float GMEM
    //
    // 设计:
    //   1. tma_store_fence()  — 保证 sC 写入对 TMA 硬件可见
    //   2. TMA Store copy     — 异步发起 (只需 tid 0)
    //   3. tma_store_wait_all() — 等待 TMA Store 完成再退出 kernel
    // -------------------------------------------------------------------------
    if (tid == 0) {
        // 构建 GMEM C tensor (FP32 row-major)
        // 注意: TMA Store 不需要 get_tma_tensor, 使用 partition_S/partition_D
        auto mC_gmem = tma_c.get_tma_tensor(make_shape(M, N));
        auto gC_tile = local_tile(
            mC_gmem,
            make_tile(Int<kTileM>{}, Int<kTileN>{}),
            make_coord(blockIdx.y, blockIdx.x));

        // TMA Store C slice (SMEM → GMEM)
        Tensor tCsC_store = tma_c.get_slice(Int<0>{}).partition_S(sC);
        Tensor tCgC_store = tma_c.get_slice(Int<0>{}).partition_D(gC_tile);

        // fence: 保证 SMEM 写入对 TMA 可见
        tma_store_fence();

        // 发起异步 TMA Store
        copy(tma_c, tCsC_store, tCgC_store);

        // commit_group: 提交当前 bulk_group (必须在 wait 之前)
        cute::tma_store_arrive();

        // 等待所有 TMA Store 完成后再退出 kernel
        tma_store_wait_all();
    }
}

// ----------------------------------------------------------------------------
// get_smem_size_pingpong_tma_store
// A_buf + B_buf + mbar(full+empty) + C_buf(float, row-major)
// ----------------------------------------------------------------------------
template <typename Config>
constexpr size_t get_smem_size_pingpong_tma_store() {
    using T = typename Config::T;
    constexpr size_t smem_bytes_A  = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(T);
    constexpr size_t smem_bytes_B  = cute::cosize(typename Config::SmemLayoutB{}) * sizeof(T);
    constexpr size_t mbar_offset   = (smem_bytes_A + smem_bytes_B + 7) & ~7UL;
    constexpr size_t mbar_bytes    = 2 * Config::kStage * sizeof(uint64_t);
    constexpr size_t smem_C_offset = (mbar_offset + mbar_bytes + 127) & ~127UL;
    // float row-major: kTileM * kTileN * 4B
    constexpr size_t smem_C_bytes  = Config::kTileM * Config::kTileN * sizeof(float);
    return smem_C_offset + smem_C_bytes;
}

// ============================================================================
// ─────────────────────────────────────────────────────────────────────────────
// 版本 B: 384 线程双 WG Cooperative GEMM (FP32 直接写回 GMEM)
// ─────────────────────────────────────────────────────────────────────────────
// ============================================================================

// ----------------------------------------------------------------------------
// load_wg_2x — Load WG (tid 256-383) 为两个 Math WG 加载 A+B0+B1
//
// A 共享 (同一 M tile), B0/B1 分别对应 WG0/WG1 的 N tile
// 一次 TMA 同时发起 3 条 TMA (A + B0 + B1), 关联同一个 mbar_full[stage]
// mbar_empty arrive_count=2: WG0 arrive 一次 + WG1 arrive 一次
// ----------------------------------------------------------------------------
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__device__ __noinline__ void load_wg_2x(
    uint64_t* __restrict__ mbar_full,
    uint64_t* __restrict__ mbar_empty,
    TmaCopyA const& tma_a,
    TmaCopyB const& tma_b,
    char* __restrict__ smem_A_ptr,
    char* __restrict__ smem_B0_ptr,
    char* __restrict__ smem_B1_ptr,
    int load_tid,
    int bx0,  // WG0 的 N tile index
    int by,   // M tile index
    int M, int N, int K)
{
    static constexpr int kStage    = Config::kStage;
    static constexpr int kTileM    = Config::kTileM;
    static constexpr int kTileN    = Config::kTileN;
    static constexpr int kTileK    = Config::kTileK;
    using T = typename Config::T;
    // 3 条 TMA: A + B0 + B1
    static constexpr int kTmaBytes = kTileM * kTileK * sizeof(T)
                                   + kTileN * kTileK * sizeof(T)
                                   + kTileN * kTileK * sizeof(T);

    if (load_tid == 0) {
        auto mA  = tma_a.get_tma_tensor(make_shape(M, K));
        auto mB  = tma_b.get_tma_tensor(make_shape(N, K));
        auto gA  = local_tile(mA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(by, _));
        auto gB0 = local_tile(mB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(bx0,   _));
        auto gB1 = local_tile(mB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(bx0+1, _));

        auto sA  = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_A_ptr)),
                               typename Config::SmemLayoutA{});
        auto sB0 = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_B0_ptr)),
                               typename Config::SmemLayoutB{});
        auto sB1 = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_B1_ptr)),
                               typename Config::SmemLayoutB{});

        Tensor tAgA  = tma_a.get_slice(Int<0>{}).partition_S(gA);
        Tensor tAsA  = tma_a.get_slice(Int<0>{}).partition_D(
            as_position_independent_swizzle_tensor(sA));
        Tensor tBgB0 = tma_b.get_slice(Int<0>{}).partition_S(gB0);
        Tensor tBsB0 = tma_b.get_slice(Int<0>{}).partition_D(
            as_position_independent_swizzle_tensor(sB0));
        Tensor tBgB1 = tma_b.get_slice(Int<0>{}).partition_S(gB1);
        Tensor tBsB1 = tma_b.get_slice(Int<0>{}).partition_D(
            as_position_independent_swizzle_tensor(sB1));

        int num_k_tiles = K / kTileK;
        int stage   = 0;
        int phase_e = 0;

        for (int k = 0; k < num_k_tiles; ++k) {
            // 等待 WG0 + WG1 都释放 stage (arrive_count=2)
            mbar_wait(&mbar_empty[stage], phase_e);

            // 3 条 TMA 关联同一个 mbar_full[stage]
            mbar_arrive_and_expect_tx(&mbar_full[stage], kTmaBytes);
            copy(tma_a.with(mbar_full[stage]), tAgA(_, _, _, k),  tAsA(_, _, _, stage));
            copy(tma_b.with(mbar_full[stage]), tBgB0(_, _, _, k), tBsB0(_, _, _, stage));
            copy(tma_b.with(mbar_full[stage]), tBgB1(_, _, _, k), tBsB1(_, _, _, stage));

            stage = (stage + 1) % kStage;
            if (stage == 0) phase_e ^= 1;
        }
    }
    // 其他 load_tid 线程: warpgroup_reg_dealloc 后空转, 无需工作
}

// ----------------------------------------------------------------------------
// math_wg_2x — 单个 Math WG Consumer (版本 B, 共用 mbar_full/empty)
// 与 pp_consumer_loop 相同逻辑, 但 arrive_count=2 (WG0 + WG1 各一次)
// Epilogue: 直接写 FP32 到 GMEM (copy(tCrC, tCgC))
// ----------------------------------------------------------------------------
template <typename Config, typename SmemTensorA, typename SmemTensorB,
          typename GmemTensorC>
__device__ __forceinline__ void math_wg_2x(
    uint64_t* __restrict__ mbar_full,
    uint64_t* __restrict__ mbar_empty,   // arrive_count=2
    SmemTensorA const& sA,
    SmemTensorB const& sB,
    GmemTensorC const& gC,    // FP32 GMEM C buffer
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
        // arrive_count=2: 每次两个 WG 各 arrive 一次
        mbar_arrive_if(&mbar_empty[prev_stage], (k > 0) && (wg_tid == 0));

        stage = (stage + 1) % kStage;
        if (stage == 0) phase_f ^= 1;
    }

    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    int last_stage = (stage - 1 + kStage) % kStage;
    mbar_arrive_if(&mbar_empty[last_stage], (num_k_tiles > 0) && (wg_tid == 0));

    // Epilogue: 直接写 FP32 到 GMEM
    copy(tCrC, tCgC);
}

// ----------------------------------------------------------------------------
// gemm_kernel_2wg_pingpong — 版本 B 主 kernel (384 线程)
//
// grid: (N/(kTileN*2), M/kTileM) — 每 block 处理 2 个相邻 N tile
//   WG0 处理 tile (blockIdx.y, blockIdx.x*2)
//   WG1 处理 tile (blockIdx.y, blockIdx.x*2+1)
//
// SMEM (kStage=2):
//   sA:  32KB (128×64×2×2B)
//   sB0: 32KB (WG0 的 B)
//   sB1: 32KB (WG1 的 B)
//   mbar: 约 64B
//   总: ≈ 96KB ✓ (FP32 输出直接写 GMEM, 无 C buffer)
// ----------------------------------------------------------------------------
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__global__ void __launch_bounds__(Config::kNumThreads2WG, 1)
gemm_kernel_2wg_pingpong(
    float*       __restrict__ Cptr,
    CUTLASS_GRID_CONSTANT TmaCopyA  const tma_a,
    CUTLASS_GRID_CONSTANT TmaCopyB  const tma_b,
    int M, int N, int K)
{
    using T = typename Config::T;
    static constexpr int kTileM = Config::kTileM;
    static constexpr int kTileN = Config::kTileN;
    static constexpr int kStage = Config::kStage;

    extern __shared__ char smem_buf[];

    // SMEM 布局: A + B0 + B1 + mbar
    constexpr size_t smem_bytes_A  = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(T);
    constexpr size_t smem_bytes_B  = cute::cosize(typename Config::SmemLayoutB{}) * sizeof(T);
    // B0 紧跟 A, B1 紧跟 B0
    constexpr size_t smem_B1_start = smem_bytes_A + smem_bytes_B;
    constexpr size_t mbar_offset   = (smem_bytes_A + smem_bytes_B * 2 + 7) & ~7UL;
    constexpr size_t mbar_bytes    = 2 * kStage * sizeof(uint64_t);

    char*     smem_A_ptr  = smem_buf;
    char*     smem_B0_ptr = smem_buf + smem_bytes_A;
    char*     smem_B1_ptr = smem_buf + smem_B1_start;
    uint64_t* mbar_full   = reinterpret_cast<uint64_t*>(smem_buf + mbar_offset);
    uint64_t* mbar_empty  = mbar_full + kStage;

    int tid = threadIdx.x;
    const bool is_load_wg = (tid >= 256);
    const bool is_wg1     = (tid >= 128 && tid < 256);
    const bool is_wg0     = (tid < 128);
    const int  wg0_tid    = tid;
    const int  wg1_tid    = tid - 128;
    const int  load_tid   = tid - 256;

    // 寄存器配额调整 (hpc-ops 关键优化)
    if (is_load_wg) {
        cutlass::arch::warpgroup_reg_dealloc<24>();
    } else {
        cutlass::arch::warpgroup_reg_alloc<168>();
    }

    // mbarrier 初始化
    // mbar_full[s]:  arrive_count=1 (Load WG 一次 TMA arrive)
    // mbar_empty[s]: arrive_count=2 (WG0 + WG1 各 arrive 一次)
    if (tid == 0) {
        for (int s = 0; s < kStage; ++s) {
            mbar_init(&mbar_full[s],  1);
            mbar_init(&mbar_empty[s], 2);
        }
        mbar_fence_init();
    }
    __syncthreads();

    // 预先 arrive: WG0 和 WG1 各 arrive 所有 mbar_empty[s]
    // 这样 mbar_empty 初始就绪 (arrive_count=2, WG0+WG1各arrive一次=共2)
    if (is_wg0 && wg0_tid == 0) {
        for (int s = 0; s < kStage; ++s) mbar_arrive(&mbar_empty[s]);
    }
    if (is_wg1 && wg1_tid == 0) {
        for (int s = 0; s < kStage; ++s) mbar_arrive(&mbar_empty[s]);
    }
    __syncthreads();

    // SMEM 张量
    auto sA  = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_A_ptr)),
                           typename Config::SmemLayoutA{});
    auto sB0 = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_B0_ptr)),
                           typename Config::SmemLayoutB{});
    auto sB1 = make_tensor(make_smem_ptr(reinterpret_cast<T*>(smem_B1_ptr)),
                           typename Config::SmemLayoutB{});

    // GMEM C tensors (FP32 直接写回)
    auto C   = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(N, _1{}));

    // Output tile 坐标
    int bx0 = blockIdx.x * 2;
    int bx1 = blockIdx.x * 2 + 1;
    int by  = blockIdx.y;
    int num_k_tiles = K / Config::kTileK;

    if (is_load_wg) {
        load_wg_2x<Config>(
            mbar_full, mbar_empty,
            tma_a, tma_b,
            smem_A_ptr, smem_B0_ptr, smem_B1_ptr,
            load_tid,
            bx0, by,
            M, N, K);
    } else if (is_wg0) {
        auto gC0 = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                              make_coord(by, bx0));
        math_wg_2x<Config>(
            mbar_full, mbar_empty,
            sA, sB0, gC0,
            wg0_tid,
            num_k_tiles);
    } else {  // is_wg1
        auto gC1 = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}),
                              make_coord(by, bx1));
        math_wg_2x<Config>(
            mbar_full, mbar_empty,
            sA, sB1, gC1,
            wg1_tid,
            num_k_tiles);
    }

    __syncthreads();
}

// ----------------------------------------------------------------------------
// get_smem_size_2wg_pingpong
// A + B0 + B1 + mbar(full+empty)
// (无 C buffer: FP32 输出直接写 GMEM)
// ----------------------------------------------------------------------------
template <typename Config>
constexpr size_t get_smem_size_2wg_pingpong() {
    using T = typename Config::T;
    constexpr size_t smem_bytes_A  = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(T);
    constexpr size_t smem_bytes_B  = cute::cosize(typename Config::SmemLayoutB{}) * sizeof(T);
    constexpr size_t mbar_offset   = (smem_bytes_A + smem_bytes_B * 2 + 7) & ~7UL;
    constexpr size_t mbar_bytes    = 2 * Config::kStage * sizeof(uint64_t);
    return mbar_offset + mbar_bytes;
}

} // namespace gemm_sm90
