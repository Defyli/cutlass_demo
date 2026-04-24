#pragma once

// ============================================================================
// GemmConfigFP8 — SM90 (Hopper) FP8 GEMM 配置
//
// 与 BF16 config.cuh 的主要差异:
//   1. 输入类型: float_e4m3_t (E4M3) 或 float_e5m2_t (E5M2)，1 字节/元素
//   2. MMA K 维度: FP8 硬件指令 K=32 (vs BF16 K=16)，所以 kTileK 至少 32
//   3. kTileK 推荐 128 (= 4 × 32)，充分利用 SM90 FP8 wgmma 吞吐
//   4. SMEM Swizzle: E4M3 1字节，ss_smem_selector 会选 Layout_K_SW128_Atom
//      (kTileK=128 时，kTileK % 64 == 0 → Swizzle<3,4,3>)
//   5. G2S 向量宽度: 128bit = 16 个 FP8 元素 (vs BF16 的 8 个)
//   6. Epilogue: FP32 累加器 → R->S->G，与 BF16 版本相同逻辑
// ============================================================================

#include "cute/tensor.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/arch/mma_sm90.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/collective/builders/sm90_common.inl"

namespace gemm_fp8_sm90 {

using namespace cute;

// ============================================================================
// GemmConfigFP8
//
// 模板参数:
//   TA_     : A 矩阵元素类型 (float_e4m3_t 或 float_e5m2_t)
//   TB_     : B 矩阵元素类型 (float_e4m3_t 或 float_e5m2_t)
//   KTileM_ : CTA M-tile 大小，必须是 64 的倍数（wgmma 最小 M=64）
//   KTileN_ : CTA N-tile 大小，必须是 8 的倍数
//   KTileK_ : CTA K-tile 大小，必须是 32 的倍数（FP8 MMA K=32）
//   KStage_ : 流水线深度
// ============================================================================
template <typename TA_,
          typename TB_,
          int KTileM_,
          int KTileN_,
          int KTileK_,
          int KStage_>
struct GemmConfigFP8 {

    using TA        = TA_;          // A 输入类型 (float_e4m3_t 等)
    using TB        = TB_;          // B 输入类型
    using AccumType = float;        // FP8 GMMA 固定 FP32 累加

    static constexpr int kTileM = KTileM_;
    static constexpr int kTileN = KTileN_;
    static constexpr int kTileK = KTileK_;
    static constexpr int kStage = KStage_;

    // FP8 MMA 指令 K 维度固定为 32
    static_assert(kTileM % 64 == 0, "kTileM must be a multiple of 64");
    static_assert(kTileN % 8  == 0, "kTileN must be a multiple of 8");
    static_assert(kTileK % 32 == 0, "kTileK must be a multiple of 32 for FP8 MMA");

    // -------------------------------------------------------------------------
    // MMA Atom: 使用 ss_op_selector 自动选择最优 FP8 wgmma Op
    //
    //   FP8 指令格式: SM90_64xNx32_F32E4M3E4M3_SS_TN
    //     - M 固定 64 (单 WarpGroup)
    //     - N 可选 8/16/.../256
    //     - K=32 (硬件约束)
    //     - SS: A/B 均从 SMEM descriptor 读取
    //     - TN: A Transposed (K-major), B Normal (K-major)
    //           即 A: (M×K) row-major, B: (N×K) row-major
    //
    //   注意: FP8 GMMA 仅支持 Major::K 布局 (K-major)
    // -------------------------------------------------------------------------
    using MMA_Op = decltype(GMMA::ss_op_selector<
        TA, TB, AccumType,
        Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>,
        GMMA::Major::K,   // A: K-major
        GMMA::Major::K    // B: K-major
    >());

    using TiledMMA = decltype(make_tiled_mma(
        MMA_Atom<MMA_Traits<MMA_Op>>{}
    ));

    // -------------------------------------------------------------------------
    // SMEM Layout: ss_smem_selector 自动选择最优 Swizzle
    //
    //   FP8 (1字节) + kTileK=128: kTileK % 64 == 0 → Layout_K_SW128_Atom
    //   等价于 Swizzle<3,4,3>，TMA 需要 128-bit swizzle (M=4)
    // -------------------------------------------------------------------------
    using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K, TA,
        Int<kTileM>, Int<kTileK>
    >());
    using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K, TB,
        Int<kTileN>, Int<kTileK>
    >());

    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{},
        make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomB{},
        make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})
    ));

    // -------------------------------------------------------------------------
    // G2S Copy (cp.async 备用，128bit/thread = 16 FP8 elements)
    // TMA kernel 不需要此配置，但留作参考和未来 cp.async 变体
    // -------------------------------------------------------------------------
    static constexpr int kVecLenAB   = 128 / (8 * sizeof(TA));  // = 16 for FP8
    static constexpr int kRowThreads = kTileK / kVecLenAB;       // = 8 for kTileK=128
    static constexpr int kColThreads = 128 / kRowThreads;        // = 16

    using G2SCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, TA>;
    using G2SCopyA = decltype(make_tiled_copy(
        G2SCopyAtom{},
        make_layout(make_shape(Int<kColThreads>{}, Int<kRowThreads>{}),
                    make_stride(Int<kRowThreads>{}, _1{})),
        make_layout(make_shape(_1{}, Int<kVecLenAB>{}))
    ));
    using G2SCopyB = G2SCopyA;

    // -------------------------------------------------------------------------
    // Epilogue Copy: R->S->G 两级写回
    //
    //   FP32 累加器 → SMEM (SmemLayoutC) → GMEM
    //   SmemLayoutC: kTileM×kTileN row-major + Swizzle<3,3,3>（FP32 bank conflict free）
    // -------------------------------------------------------------------------
    using SmemLayoutAtomC = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<kTileM>{}, Int<kTileN>{}),
                    make_stride(Int<kTileN>{}, _1{}))
    ));
    using SmemLayoutC = decltype(tile_to_shape(
        SmemLayoutAtomC{},
        make_shape(Int<kTileM>{}, Int<kTileN>{})
    ));

    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, AccumType>;

    static constexpr int kVecLenC    = 128 / (8 * sizeof(AccumType)); // = 4 for FP32
    static constexpr int kColThreadsC = kTileN / kVecLenC;
    static constexpr int kRowThreadsC = 128 / kColThreadsC;
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, AccumType>;
    using S2GCopyC = decltype(make_tiled_copy(
        S2GCopyAtomC{},
        make_layout(make_shape(Int<kRowThreadsC>{}, Int<kColThreadsC>{}),
                    make_stride(Int<kColThreadsC>{}, _1{})),
        make_layout(make_shape(_1{}, Int<kVecLenC>{}))
    ));

    // 单 WarpGroup: 128 线程
    static constexpr int kNumThreads = 128;
    // Ping-Pong kernel: Consumer WG (128 线程) + Producer 1 warp (32 线程) = 160 线程
    static constexpr int kNumThreadsPP = 160;
};

// ============================================================================
// SMEM 大小计算
// ============================================================================

// TMA FP8 kernel: A_buf + B_buf + kStage 个 mbarrier
template <typename Config>
constexpr size_t get_smem_size_fp8_tma() {
    constexpr size_t smem_A =
        cute::cosize(typename Config::SmemLayoutA{}) * sizeof(typename Config::TA);
    constexpr size_t smem_B =
        cute::cosize(typename Config::SmemLayoutB{}) * sizeof(typename Config::TB);
    constexpr size_t smem_AB  = smem_A + smem_B;
    constexpr size_t mbar_off = (smem_AB + 7) & ~7;
    constexpr size_t mbar_sz  = Config::kStage * sizeof(uint64_t);
    return mbar_off + mbar_sz;
}

// Epilogue buffer (复用 smem，取 AB 与 C 的最大值)
template <typename Config>
constexpr size_t get_smem_size_fp8_tma_with_epilogue() {
    constexpr size_t smem_AB  = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(typename Config::TA)
                               + cute::cosize(typename Config::SmemLayoutB{}) * sizeof(typename Config::TB);
    constexpr size_t smem_C   = cute::cosize(typename Config::SmemLayoutC{}) * sizeof(typename Config::AccumType);
    constexpr size_t smem_main = smem_AB > smem_C ? smem_AB : smem_C;
    constexpr size_t mbar_off = (smem_AB + 7) & ~7;
    constexpr size_t mbar_sz  = Config::kStage * sizeof(uint64_t);
    // 取最大: main buf 还是 mbar_off + mbar_sz
    constexpr size_t total_mbar = mbar_off + mbar_sz;
    return smem_main > total_mbar ? smem_main : total_mbar;
}

} // namespace gemm_fp8_sm90
