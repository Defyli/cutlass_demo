#pragma once

// ============================================================================
// mbarrier PTX 封装
//
// mbarrier 是 Hopper (SM90) 引入的 phase-based 双向屏障:
//   state = phase(1bit) | arrive_count(20bit) | tx_count(21bit)
//   当 arrive_count==0 AND tx_count==0 时, phase 翻转 (0->1->0->...)
//   TMA 完成时硬件自动将 tx_count 减至 0
//
// 典型用法:
//   Producer:
//     mbar_arrive_and_expect_tx(mbar, N_bytes)  // 宣告将传输 N_bytes
//     TMA copy(... with(mbar) ...)               // 发起异步传输
//   Consumer:
//     mbar_wait(mbar, phase)                     // 等传输完成
// ============================================================================

#include "cute/arch/util.hpp"    // cast_smem_ptr_to_uint

namespace gemm_sm90 {

// 初始化 mbarrier, arrive_count 指定需要多少次 arrive 才能翻转 phase
CUTE_DEVICE void mbar_init(uint64_t* mbar, int arrive_count) {
    uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(smem_ptr), "r"(arrive_count));
}

// arrive + 预告 TMA 字节数
// TMA 完成后硬件自动减少 tx_count; tx_count 归零且 arrive_count 归零后 phase 翻转
CUTE_DEVICE void mbar_arrive_and_expect_tx(uint64_t* mbar, int tx_bytes) {
    uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(smem_ptr), "r"(tx_bytes));
}

// fence: 确保 mbarrier.init 对整个 cluster 可见
// 必须在 mbar_init 之后、第一次 __syncthreads() 之前调用
CUTE_DEVICE void mbar_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" :::);
}

// 等待 mbarrier 达到 expected_phase
//
// 使用 mbarrier.try_wait.parity.acquire.cta:
//   - acquire 语义: 保证 TMA 写入的数据在 wait 返回后对当前 CTA 可见 (内存序)
//     注意: 此内存可见性只对"执行该指令的线程"有效, 不会自动传播到其他线程!
//     因此 Consumer 的全 128 线程都必须执行此 wait.
//   - 失败时使用 nanosleep(0x100) 退避 (~256ns), 让 warp scheduler 切换其他 warp,
//     避免 busy-poll 占用 SM issue slot
CUTE_DEVICE void mbar_wait(uint64_t* mbar, int expected_phase) {
    uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(mbar);
    asm volatile(
        "{\n"
        ".reg .pred complete;\n"
        "MBAR_WAIT_%=:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 complete, [%0], %1;\n"
        "@complete bra MBAR_DONE_%=;\n"
        "nanosleep.u32 0x100;\n"
        "bra MBAR_WAIT_%=;\n"
        "MBAR_DONE_%=:\n"
        "}\n"
        :: "r"(smem_ptr), "r"(expected_phase) : "memory");
}

// 纯计数 arrive (不携带 tx_count)
// 用于 Consumer 通知 Producer: "我已完成 wgmma, 这个 stage 的 SMEM 可以覆盖"
CUTE_DEVICE void mbar_arrive(uint64_t* mbar) {
    uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(mbar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n"
                 :: "r"(smem_ptr));
}

// Predicated arrive: 等价于 if (pred) mbar_arrive(mbar)
// 用 PTX @pred 指令实现, 避免 C/C++ if-else 在 wgmma 上下文中产生 divergent path
// (C7520 warning: wgmma serialized due to compiler-inserted WG.AR in divergent path)
CUTE_DEVICE void mbar_arrive_if(uint64_t* mbar, bool pred) {
    uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(mbar);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.u32 p, %1, 0;\n"
        "@p mbarrier.arrive.shared::cta.b64 _, [%0];\n"
        "}\n"
        :: "r"(smem_ptr), "r"((uint32_t)pred));
}

// ============================================================================
// Cluster-scope mbarrier 操作 (用于 Cluster TMA Multicast)
// ============================================================================

// arrive + 预告 TMA 字节数 (远端 CTA, cluster 作用域)
//
// 使用 shared::cluster 地址空间, 通过 mapa 指令将本 CTA 的 smem 地址映射到远端 CTA
// 用于 Cluster TMA Multicast 场景: leader CTA (如 cta_rank=0) 负责 arrive 所有 CTA 的 mbar
// 
// cta_id: cluster 内的目标 CTA ID (0, 1, ... cluster_size-1)
// pred:   是否执行该 arrive (1=执行, 0=跳过)
//
// 典型用法:
//   if (prod_tid == 0 && cta_rank == 0) {  // leader producer thread
//     mbar_arrive_and_expect_tx(mbar_full[s], tx_bytes);          // local CTA
//     mbar_arrive_and_expect_tx_remote(mbar_full[s], tx_bytes, 1, 1); // remote CTA1
//   }
CUTE_DEVICE void mbar_arrive_and_expect_tx_remote(
    uint64_t* mbar, int tx_bytes, int cta_id, int pred = 1)
{
    uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(mbar);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        ".reg .b32 remAddr32;\n"
        "setp.eq.u32 p, %2, 1;\n"
        "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n"
        "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], %3;\n"
        "}\n"
        :: "r"(smem_ptr), "r"(cta_id), "r"(pred), "r"(tx_bytes)
        : "memory");
}

} // namespace gemm_sm90
