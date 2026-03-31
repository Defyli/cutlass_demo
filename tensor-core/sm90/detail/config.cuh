#pragma once

// ============================================================================
// GemmConfig + SMEM size helpers
//
// 包含:
//   - GemmConfig<T, kTileM, kTileN, kTileK, kStage>
//       定义 MMA atom / SMEM layout / G2S copy / 线程数等编译期常量
//   - get_smem_size_tma<Config>()
//   - get_smem_size_cp_async<Config>()
//   - get_smem_size_pingpong<Config>()
// ============================================================================

#include "cute/tensor.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/arch/mma_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/collective/builders/sm90_common.inl"

namespace gemm_sm90 {

using namespace cute;

// ============================================================================
// GemmConfig
//
// 关键设计差异 (vs Ada SM80/89):
//
//  特性             Ada (SM80/89)              Hopper (SM90)
//  ---------------------------------------------------------------
//  MMA 指令        mma.sync (32线程/Warp)     wgmma.mma_async (128线程/WarpGroup)
//  MMA 形状        16x8x16                    64xNx16 (N=8,16,...,256)
//  G->S 搬运       cp.async (所有线程分工)     TMA (1线程发起整块)
//  同步原语        cp_async_fence/wait         mbarrier (phase-based)
//  SMEM->Reg      ldmatrix 显式搬运            SS模式: wgmma 直接读 SMEM, 不需要
//  SMEM Swizzle   Swizzle<3,3,3>              Swizzle<3,4,3> (M参数=4)
//  累加器          F16 或 F32                  F32 (BF16/FP16输入固定F32累加)
// ============================================================================
template <typename T_,          // 输入类型 (bfloat16_t 或 half_t)
          int KTileM_,          // CTA M-tile 大小, 必须是 64 的倍数
          int KTileN_,          // CTA N-tile 大小, 必须是 8 的倍数
          int KTileK_,          // CTA K-tile 大小, 必须是 16 的倍数
          int KStage_>          // Pipeline stages
struct GemmConfig {

    using T         = T_;
    using AccumType = float;    // Hopper BF16/FP16 GMMA 固定用 FP32 累加

    static constexpr int kTileM = KTileM_;
    static constexpr int kTileN = KTileN_;
    static constexpr int kTileK = KTileK_;
    static constexpr int kStage = KStage_;

    static_assert(kTileM % 64 == 0, "kTileM must be multiple of 64");
    static_assert(kTileN % 8  == 0, "kTileN must be multiple of 8");
    static_assert(kTileK % 16 == 0, "kTileK must be multiple of 16");

    // -------------------------------------------------------------------------
    // MMA Atom: 由 GMMA::ss_op_selector 根据 (ElementA, ElementB, ElementC,
    //   TileShape_MNK) 自动选择最优的 SM90_64xNxK_... Op 类型。
    //
    //   SS = Shared-Shared: A 从 SMEM 读, B 从 SMEM 读。
    //   M 固定=64, N 可选 8/16/32/64/96/128/192/256。
    //
    //   CRITICAL: SM90 GMMA 不应使用 AtomLayout 堆叠!
    //     单个 SM90_64xNx16 atom 已是完整的 128线程 warpgroup MMA.
    //     AtomLayout 堆叠会导致 partition_C 错误映射，部分输出无法写回.
    // -------------------------------------------------------------------------
    using MMA_Op = decltype(GMMA::ss_op_selector<
        T, T, AccumType,
        Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>,
        GMMA::Major::K,   // A: K-major (行主序)
        GMMA::Major::K    // B: K-major (行主序)
    >());

    using TiledMMA = decltype(make_tiled_mma(
        MMA_Atom<MMA_Traits<MMA_Op>>{}
    ));

    // -------------------------------------------------------------------------
    // SMEM Layout: 由 ss_smem_selector 根据 Tile 尺寸自动选择最优 Swizzle。
    //
    //   Major::K 布局下，按 kTileK 选择:
    //     kTileK % 64==0 → Layout_K_SW128_Atom (Swizzle<3,4,3>, 最优)
    //     kTileK % 32==0 → Layout_K_SW64_Atom
    //     kTileK % 16==0 → Layout_K_SW32_Atom
    //     else            → Layout_K_INTER_Atom (无 swizzle, bank conflict 多)
    //
    //   TMA 要求: swizzle 必须是 0/32/64/128 bit 中的一种，且 M 参数=4。
    // -------------------------------------------------------------------------
    using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K, T,
        Int<kTileM>, Int<kTileK>
    >());
    using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K, T,
        Int<kTileN>, Int<kTileK>
    >());

    // 扩展为 (kTileM, kTileK, kStage) 的完整 SMEM layout
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{},
        make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomB{},
        make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})
    ));

    // -------------------------------------------------------------------------
    // G2S Copy (cp.async 版本, 用于不带TMA的参考kernel)
    //   128线程, 每次 128bit (8个BF16/FP16)
    // -------------------------------------------------------------------------
    static constexpr int kVecLen     = 128 / (8 * sizeof(T));
    static constexpr int kRowThreads = kTileK / kVecLen;
    static constexpr int kColThreads = 128 / kRowThreads;

    using G2SCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, T>;
    using G2SCopyA = decltype(make_tiled_copy(
        G2SCopyAtom{},
        make_layout(make_shape(Int<kColThreads>{}, Int<kRowThreads>{}),
                    make_stride(Int<kRowThreads>{}, _1{})),
        make_layout(make_shape(_1{}, Int<kVecLen>{}))
    ));
    using G2SCopyB = G2SCopyA;

    // 单 WarpGroup kernel: 128 线程
    static constexpr int kNumThreads   = 128;
    // Ping-Pong kernel (v5/v6): Consumer WG (128 线程) + Producer 1 warp (32 线程) = 160 线程
    //
    // 寄存器隔离设计 (v5 核心):
    //   v4 问题: Consumer(__forceinline__) + Producer(__forceinline__)
    //     全部内联 → ptxas 为 if/else 两分支都分配寄存器 → 232 reg × 160 = 37120
    //     floor(65536/37120) = 1 block → occupancy=1 ❌
    //
    //   v5 方案: Consumer __forceinline__ + Producer __noinline__
    //     - Consumer 执行 wgmma → 必须 __forceinline__ 避免 C7510
    //     - Producer 只做 TMA  → 可以 __noinline__, 不影响 wgmma pipeline
    //     - TMA 张量视图 (tAgA/tAsA/tBgB/tBsB) 移入 Producer 函数内部构建
    //     - kernel 本体 Consumer 路径只有 gC/sA/sB + wgmma 累加器
    //     预期: ~154 reg × 160 = 24640 reg/block
    //     SM 可驻留: floor(65536/24640) = 2 blocks ✅
    //     kStage=3: SMEM=96KB → floor(227/96) = 2 blocks → occupancy=2 ✅
    static constexpr int kNumThreadsPP = 160;

    // 双 WG Cooperative kernel (v6/2WG): 2 Math WG + 1 Load WG = 384 线程
    //
    //   基于 hpc-ops 学习的两个新优化:
    //   1. warpgroup_reg_alloc<168> / warpgroup_reg_dealloc<24>
    //      Math WG: 增加到 168 reg (充分供 wgmma 累加器使用)
    //      Load WG: 减少到 24 reg  (只做 TMA 控制, 不需要大量寄存器)
    //
    //   2. 每个 block 同时处理 2 个相邻 N tile:
    //      WG0 → tile (by, bx*2),   WG1 → tile (by, bx*2+1)
    //      grid.x = N/(kTileN*2)    (grid.x 减半, 每 block 处理 2 tiles)
    //      sA 共享 (同一 M tile), sB0/sB1 分别对应两个 N tile
    //      → 充分利用 2 Math WG 的并行计算能力
    //
    //   SMEM (kStage=2): A(32) + B0(32) + B1(32) + C0(32) + C1(32) ≈ 160KB ✓
    //   TMA Store Epilogue: FP32→BF16 + STMATRIX + TMA Store (异步写回 GMEM)
    //
    //   __launch_bounds__(384, 1): 384 线程/block, 1 block/SM 目标
    //   注意: 384×168 = 64512 ≈ 65536, SM 资源接近上限, occupancy≈1
    static constexpr int kNumThreads2WG = 384;
};

// ============================================================================
// SMEM 大小计算辅助函数
// ============================================================================

// TMA kernel: A_buf + B_buf + kStage 个 mbarrier (8B each)
template <typename Config>
constexpr size_t get_smem_size_tma() {
    constexpr size_t smem_AB =
        (cute::cosize(typename Config::SmemLayoutA{}) +
         cute::cosize(typename Config::SmemLayoutB{})) * sizeof(typename Config::T);
    constexpr size_t mbar_off = (smem_AB + 7) & ~7;
    constexpr size_t mbar_sz  = Config::kStage * sizeof(uint64_t);
    return mbar_off + mbar_sz;
}

// cp.async kernel: A_buf + B_buf (无 mbarrier)
template <typename Config>
constexpr size_t get_smem_size_cp_async() {
    return (cute::cosize(typename Config::SmemLayoutA{}) +
            cute::cosize(typename Config::SmemLayoutB{})) * sizeof(typename Config::T);
}

// Ping-Pong kernel: A_buf + B_buf + 2×kStage 个 mbarrier (mbar_full + mbar_empty)
template <typename Config>
constexpr size_t get_smem_size_pingpong() {
    constexpr size_t smem_AB =
        (cute::cosize(typename Config::SmemLayoutA{}) +
         cute::cosize(typename Config::SmemLayoutB{})) * sizeof(typename Config::T);
    constexpr size_t mbar_off = (smem_AB + 7) & ~7;
    constexpr size_t mbar_sz  = 2 * Config::kStage * sizeof(uint64_t);
    return mbar_off + mbar_sz;
}

} // namespace gemm_sm90
