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
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cutlass/numeric_types.h"

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
    // MMA Atom: SM90_64x{N}x16_F32BF16BF16_SS
    //   SS = Shared-Shared: A从SMEM读, B从SMEM读
    //   ThrID = Layout<_128>: 整个WarpGroup (128线程) 参与一次wgmma
    //   M固定=64, N可选8/16/32/64/96/128/192/256, K固定=16
    //
    //   CRITICAL: SM90 GMMA 不应使用 AtomLayout 堆叠!
    //     单个 SM90_64xNx16 atom 已是完整的 128线程 warpgroup MMA.
    //     AtomLayout 堆叠会导致 partition_C 错误映射，部分输出无法写回.
    // -------------------------------------------------------------------------
    using MMA_Op = SM90_64x128x16_F32BF16BF16_SS<
        GMMA::Major::K,    // A: K-major (行主序)
        GMMA::Major::K>;   // B: K-major (行主序)

    using TiledMMA = decltype(make_tiled_mma(
        MMA_Atom<MMA_Traits<MMA_Op>>{}
    ));

    // -------------------------------------------------------------------------
    // SMEM Layout: GMMA 硬件要求的 K-major + 128bit Swizzle
    //
    //   GMMA::Layout_K_SW128_Atom<T> 等价于:
    //     ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<(8,64_bf16),(64,1)>>
    //   TMA 要求: swizzle 必须是 0/32/64/128 bit 中的一种, 且 M=4
    // -------------------------------------------------------------------------
    using SmemLayoutAtomA = GMMA::Layout_K_SW128_Atom<T>;
    using SmemLayoutAtomB = GMMA::Layout_K_SW128_Atom<T>;

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
    // Ping-Pong kernel: 2 个 WarpGroup (Producer + Consumer) = 256 线程
    static constexpr int kNumThreadsPP = 256;
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
