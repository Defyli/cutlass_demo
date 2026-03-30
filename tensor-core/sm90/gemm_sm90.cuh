#pragma once

// ============================================================================
// SM90 (Hopper) CuTe GEMM — 主入口头文件
//
// 输入:  BF16 (A: MxK row-major, B: NxK row-major)
// 输出:  FP32 (C: MxN row-major)
// 计算:  C = A * B^T
//
// 包含的 kernel:
//   gemm_kernel_tma      — 单WG, TMA + GMMA (SS模式)
//   gemm_kernel_cp_async — 单WG, cp.async + GMMA (对比用)
//   gemm_kernel_pingpong — 双WG Ping-Pong, TMA + GMMA
//
// 文件结构:
//   gemm_sm90.cuh              ← 本文件，对外唯一入口
//   detail/config.cuh          ← GemmConfig + SMEM size helpers
//   detail/mbarrier.cuh        ← mbar_* PTX 封装 (init/arrive/wait)
//   detail/kernel_tma.cuh      ← gemm_kernel_tma
//   detail/kernel_cp_async.cuh ← gemm_kernel_cp_async
//   detail/kernel_pingpong.cuh ← pp_producer_loop + pp_consumer_loop
//                                 + gemm_kernel_pingpong
// ============================================================================

#include "detail/config.cuh"
#include "detail/mbarrier.cuh"
#include "detail/kernel_tma.cuh"
#include "detail/kernel_cp_async.cuh"
#include "detail/kernel_pingpong.cuh"
