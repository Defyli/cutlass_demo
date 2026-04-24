#pragma once

// ============================================================================
// SM90 (Hopper) FP8 CuTe GEMM — 主入口头文件
//
// 输入:  FP8 E4M3 (A: MxK row-major, B: NxK row-major)
// 输出:  FP32 (C: MxN row-major)
// 计算:  C = A * B^T
//
// 包含的 kernel:
//   gemm_kernel_fp8_tma         — 单WG, TMA + WGMMA SS 模式 (v1/v3)
//   gemm_kernel_fp8_tma_swizzle — 单WG + block swizzle (v4)
//   gemm_kernel_fp8_pingpong    — Consumer WG (128T) + Producer (32T) (v5)
//   gemm_kernel_fp8_persistent  — Persistent PP kernel, 消除 wave quantization (v7)
//
// 文件结构:
//   gemm_fp8_sm90.cuh                ← 本文件，对外唯一入口
//   detail/config_fp8.cuh            ← GemmConfigFP8 + SMEM size helpers
//   detail/mbarrier.cuh              ← mbar_* PTX 封装 (FP8 专用副本)
//   detail/kernel_fp8_tma.cuh        ← gemm_kernel_fp8_tma / _swizzle
//   detail/kernel_fp8_pingpong.cuh   ← gemm_kernel_fp8_pingpong (v5)
//   detail/kernel_fp8_persistent.cuh ← gemm_kernel_fp8_persistent (v7)
// ============================================================================

#include "detail/config_fp8.cuh"
#include "detail/mbarrier.cuh"
#include "detail/kernel_fp8_tma.cuh"
#include "detail/kernel_fp8_pingpong.cuh"
#include "detail/kernel_fp8_persistent.cuh"
