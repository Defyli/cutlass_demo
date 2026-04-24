#pragma once

// ============================================================================
// Kernel 2: gemm_kernel_cp_async  (单 WarpGroup, cp.async + WGMMA-SS)
//
// 设计:
//   128 线程 (1 WarpGroup), cp.async 搬运 + wgmma SS 模式计算
//
// SS 模式说明 (SM90 特性):
//   SM90 wgmma SS (Shared-Shared): A/B 均从 SMEM descriptor 直接读取，
//   硬件内部维护 SMEM → 计算单元的异步流水，无需程序员显式 ldmatrix。
//   fragment_A/B 是 DescriptorIterator，不是真实寄存器，
//   编译器 (ptxas) 自动安排最优的 S->R 流水窗口。
//
//   对比 SM89 RS 模式:
//     SM89: 必须显式 ldmatrix 做 S->R，才能控制流水重叠
//     SM90 SS: 硬件级流水，显式 ldmatrix 反而多占寄存器 (198→220)，无收益
//
// 流水线: G->S 多级缓冲 (cp.async + cp_async_fence/wait)
//   Prologue: 预取 kStage-1 个 tile 到 SMEM
//   主循环: 每轮 发射下一 tile 的 cp.async，同时 wgmma 计算当前 tile
//   等待: cp_async_wait<kStage-2> 保证下一 tile 数据就绪再切换 stage
//
// Epilogue: R->S->G 两级写回
//   1. tCrC (FP32 寄存器) → sC (SMEM, R2S via UniversalCopy<int>)
//   2. sC (SMEM) → gC (GMEM, S2G via 128bit 向量化)
//   两级设计避免 FP32 累加器寄存器直接写 GMEM 的不连续访存
//
// SMEM: 96 KB (kStage=3)
// 寄存器: ~198 (SS 模式, 无额外 A fragment 缓冲)
// ============================================================================

#include "detail/config.cuh"

namespace gemm_sm90 {

template <typename Config>
__global__ void __launch_bounds__(Config::kNumThreads, 1)
gemm_kernel_cp_async(
    float*       __restrict__ Cptr,
    const typename Config::T* __restrict__ Aptr,
    const typename Config::T* __restrict__ Bptr,
    int M, int N, int K)
{
    using T       = typename Config::T;
    using AccumT  = typename Config::AccumType;
    static constexpr int kTileM = Config::kTileM;
    static constexpr int kTileN = Config::kTileN;
    static constexpr int kTileK = Config::kTileK;
    static constexpr int kStage = Config::kStage;

    // ------------------------------------------------------------------
    // 共享内存布局:
    //   前半段: sA (kStage) + sB (kStage)  — 用于 G->S 多级缓冲
    //   后半段（复用）: sC (kTileM × kTileN) — 用于 Epilogue R->S->G
    //   sA/sB 与 sC 的生命周期不重叠，安全复用同一块 SMEM
    // ------------------------------------------------------------------
    extern __shared__ char smem_buf[];
    constexpr size_t smem_bytes_A = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(T);

    T*      smem_A_ptr = reinterpret_cast<T*>(smem_buf);
    T*      smem_B_ptr = reinterpret_cast<T*>(smem_buf + smem_bytes_A);
    AccumT* smem_C_ptr = reinterpret_cast<AccumT*>(smem_buf);  // 复用 sA 起始地址（与 sA/sB 生命周期不重叠）

    auto sA = make_tensor(make_smem_ptr(smem_A_ptr), typename Config::SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(smem_B_ptr), typename Config::SmemLayoutB{});

    int tid = threadIdx.x;
    int bx  = blockIdx.x;
    int by  = blockIdx.y;

    // ------------------------------------------------------------------
    // Gmem tensors
    // ------------------------------------------------------------------
    auto gA_full = make_tensor(make_gmem_ptr(Aptr), make_shape(M, K), make_stride(K, _1{}));
    auto gB_full = make_tensor(make_gmem_ptr(Bptr), make_shape(N, K), make_stride(K, _1{}));
    auto gC_full = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(N, _1{}));

    auto gA = local_tile(gA_full, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(by, _));
    auto gB = local_tile(gB_full, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(bx, _));
    auto gC = local_tile(gC_full, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(by, bx));

    // ------------------------------------------------------------------
    // G2S copy (cp.async, 128bit/thread)
    // ------------------------------------------------------------------
    typename Config::G2SCopyA g2s_copy_a;
    typename Config::G2SCopyB g2s_copy_b;
    auto g2s_thr_a = g2s_copy_a.get_slice(tid);
    auto g2s_thr_b = g2s_copy_b.get_slice(tid);

    auto tAgA = g2s_thr_a.partition_S(gA);   // (CPY, CPY_M, CPY_K, num_k_tiles)
    auto tAsA = g2s_thr_a.partition_D(sA);   // (CPY, CPY_M, CPY_K, kStage)
    auto tBgB = g2s_thr_b.partition_S(gB);   // (CPY, CPY_N, CPY_K, num_k_tiles)
    auto tBsB = g2s_thr_b.partition_D(sB);   // (CPY, CPY_N, CPY_K, kStage)

    // ------------------------------------------------------------------
    // MMA setup (SS 模式: A/B 均从 SMEM descriptor 读取)
    // ------------------------------------------------------------------
    typename Config::TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    Tensor tCsA = thr_mma.partition_A(sA);           // (MMA, MMA_M, MMA_K, kStage)
    Tensor tCsB = thr_mma.partition_B(sB);           // (MMA, MMA_N, MMA_K, kStage)
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);     // SS 模式: DescriptorIterator
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);     // SS 模式: DescriptorIterator
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);     // FP32 寄存器累加器
    clear(tCrC);

    int num_k_tiles = K / kTileK;

    // ------------------------------------------------------------------
    // Prologue: G->S 预取 kStage-1 个 tile
    // ------------------------------------------------------------------
    int prefetch = cute::min(kStage - 1, num_k_tiles);
    for (int s = 0; s < prefetch; ++s) {
        copy(g2s_copy_a, tAgA(_, _, _, s), tAsA(_, _, _, s));
        copy(g2s_copy_b, tBgB(_, _, _, s), tBsB(_, _, _, s));
        cp_async_fence();
    }
    cp_async_wait<kStage - 2>();
    __syncthreads();

    int read_stage  = 0;
    int write_stage = prefetch % kStage;

    // ------------------------------------------------------------------
    // Main Loop: G->S 流水 + wgmma SS 计算
    //
    // 每轮:
    //   1. 发射下一 tile 的 cp.async (ik==0 时，与 wgmma 重叠)
    //   2. cp_async_wait<kStage-2> + sync: 等待下一 stage 就绪
    //   3. wgmma 计算当前 tile (SS 模式，硬件自动处理 SMEM→计算)
    //   4. 切换 read_stage
    // ------------------------------------------------------------------
    for (int itile = 0; itile < num_k_tiles; ++itile) {
        // G->S 预取下一 tile
        int next_k = itile + (kStage - 1);
        if (next_k < num_k_tiles) {
            copy(g2s_copy_a, tAgA(_, _, _, next_k), tAsA(_, _, _, write_stage));
            copy(g2s_copy_b, tBgB(_, _, _, next_k), tBsB(_, _, _, write_stage));
            cp_async_fence();
            write_stage = (write_stage + 1) % kStage;
        }

        // wgmma 计算当前 tile (SS 模式，遍历 tile 内所有 K-slice)
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        gemm(tiled_mma,
             tCrA(_, _, _, read_stage),
             tCrB(_, _, _, read_stage),
             tCrC);
        warpgroup_commit_batch();

        // 等待下一 stage 就绪（与当前 wgmma 重叠）
        cp_async_wait<kStage - 2>();
        __syncthreads();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrC);

        read_stage = (read_stage + 1) % kStage;
    }

    // ------------------------------------------------------------------
    // Epilogue: R->S->G 两级写回
    //
    // 第一级 R->S: tCrC (FP32 寄存器累加器) → sC (SMEM)
    // 第二级 S->G: sC (SMEM) → gC (GMEM, 128bit 向量化)
    // ------------------------------------------------------------------
    __syncthreads();

    auto sC = make_tensor(make_smem_ptr(smem_C_ptr), typename Config::SmemLayoutC{});

    // R->S
    auto r2s_copy_c = make_tiled_copy_C(typename Config::R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_c  = r2s_copy_c.get_slice(tid);
    auto tCrC_r2s   = r2s_thr_c.retile_S(tCrC);
    auto tCsC_r2s   = r2s_thr_c.partition_D(sC);
    copy(r2s_copy_c, tCrC_r2s, tCsC_r2s);

    __syncthreads();

    // S->G
    typename Config::S2GCopyC s2g_copy_c;
    auto s2g_thr_c = s2g_copy_c.get_slice(tid);
    auto tCsC_s2g  = s2g_thr_c.partition_S(sC);
    auto tCgC_s2g  = s2g_thr_c.partition_D(gC);
    copy(s2g_copy_c, tCsC_s2g, tCgC_s2g);
}

} // namespace gemm_sm90
