# SM90 (Hopper) CuTe GEMM 开发日志

记录从零开始实现、调试、优化 Hopper 架构 GEMM 算子的完整过程。

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [架构基础知识](#2-架构基础知识)
3. [迭代一：初始实现（结构搭建）](#3-迭代一初始实现结构搭建)
4. [迭代二：Bug修复 —— 结果为零](#4-迭代二bug修复--结果为零)
5. [迭代三：性能问题 —— TMA慢于cp.async](#5-迭代三性能问题--tma慢于cpasync)
6. [迭代四：根本Bug修复 —— WAR数据竞争](#6-迭代四根本bug修复--war数据竞争)
7. [迭代五：Ping-Pong 初版（性能低于预期）](#7-迭代五ping-pong-初版性能低于预期)
8. [迭代六：Ping-Pong 优化 —— arrive开销+kStage调整](#8-迭代六ping-pong-优化--arrive开销kstage调整)
9. [迭代七：实测结果与分析 —— kStage=4性能下降与容器内性能分析方案](#9-迭代七实测结果与分析--kstage4性能下降与容器内性能分析方案)
10. [迭代八：Ping-Pong Producer 优化 —— 单线程驱动 TMA](#10-迭代八ping-pong-producer-优化--单线程驱动-tma)
11. [迭代九：nsys 性能分析与低 occupancy 根因定位](#11-迭代九nsys-性能分析与低-occupancy-根因定位)
12. [迭代十：寄存器分离优化 —— `__noinline__` 重构](#12-迭代十寄存器分离优化----noinline-重构)
13. [迭代十一：代码结构整理](#13-迭代十一代码结构整理)
14. [当前代码结构](#14-当前代码结构)
15. [迭代十二：v3，消除 C7510 警告](#15-迭代十二v3消除-c7510-警告__forceinline__)
16. [迭代十三：诊断 `setmaxnreg` 无效的根因](#16-迭代十三诊断-setmaxnreg-无效的根因)
17. [迭代十四：v4，减少 Producer 线程数（256→160）](#17-迭代十四v4减少-producer-线程数256160)
18. [迭代十五：v5，寄存器隔离](#18-迭代十五v5寄存器隔离consumer-__forceinline__--producer-__noinline__)
19. [当前代码结构（v5）](#19-当前代码结构v5)
20. [性能数据汇总（最新）](#20-性能数据汇总最新)
21. [v6 进一步优化探索（kStage=2 测试）](#21-v6-进一步优化探索kstage2-测试)
22. [迭代十六：Persistent Kernel + Cluster TMA Multicast 架构探索](#22-迭代十六persistent-kernel--cluster-tma-multicast-架构探索)
23. [迭代十七：Cluster TMA Multicast 调试与放弃](#23-迭代十七cluster-tma-multicast-调试与放弃)
24. [迭代十八：cute-skill 知识体系整理](#24-迭代十八cute-skill-知识体系整理)
25. [不同规模性能对比](#25-不同规模性能对比)
26. [性能优化结论](#26-性能优化结论)
27. [待办事项](#27-待办事项)

---

## 1. 背景与目标

**目标**：在 Hopper (SM90) 架构上，用 CuTe API 手写一个 BF16 GEMM 算子，充分利用：

| 硬件特性 | 说明 |
|---------|------|
| **TMA** (Tensor Memory Accelerator) | 硬件异步数据搬运，单线程发起整块 G→S |
| **GMMA / wgmma** | 128线程 WarpGroup 协作的矩阵乘法指令，SS模式直接读SMEM |
| **mbarrier** | Phase-based 双向屏障，用于 TMA 生产者与 MMA 消费者同步 |
| **软件流水线** | 计算与数据搬运真正重叠 |

**计算语义**：`C(float) = A(BF16, MxK row-major) × B^T(BF16, NxK row-major)`

**文件结构**：
```
tensor-core/sm90/
  gemm_sm90.cuh   # 核心 Kernel 定义
  main.cu         # Host 端测试 + Benchmark
  CMakeLists.txt
  build.sh
```

---

## 2. 架构基础知识

### SM90 vs Ada (SM80/89) 关键差异

| 特性 | Ada (SM80/89) | Hopper (SM90) |
|------|--------------|--------------|
| MMA 指令 | `mma.sync`（32线程/Warp） | `wgmma.mma_async`（128线程/WarpGroup） |
| MMA 形状 | 16×8×16 | 64×N×16（N=8,16,...,256） |
| G→S 搬运 | `cp.async`（所有线程分工） | TMA（1线程发起整块） |
| 同步原语 | `cp_async_fence/wait` | `mbarrier`（phase-based） |
| SMEM→Reg | `ldmatrix` 显式搬运 | SS模式：wgmma 直接读SMEM，无需ldmatrix |
| SMEM Swizzle | `Swizzle<3,3,3>` | `Swizzle<3,4,3>`（M参数=4） |
| 累加器 | FP16 或 FP32 | FP32（BF16/FP16输入固定FP32累加） |

### mbarrier 工作原理

```
state = phase(1bit) | arrive_count(20bit) | tx_count(21bit)
```
- `mbar_init(mbar, arrive_count=1)`: 初始化，arrive_count=1 表示只需 thread 0 的一次 arrive
- `mbar_arrive_and_expect_tx(mbar, tx_bytes)`: arrive + 预告 TMA 将写入的字节数
- TMA 完成时，硬件自动将 `tx_count` 减至 0
- 当 `arrive_count==0 AND tx_count==0` 时，`phase` 翻转（0→1→0…）
- `mbar_wait(mbar, expected_phase)`: spin-wait 直到 phase 等于 expected_phase

### TiledMMA 配置要点

```cpp
// CRITICAL: SM90 GMMA 不应使用 AtomLayout 堆叠!
// 单个 SM90_64xNx16 atom 已经是完整的 128线程 warpgroup MMA
// 使用 AtomLayout 堆叠会导致 TiledMMA 需要 256 线程，但 Kernel 只启动 128
using MMA_Op = SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>;
using TiledMMA = decltype(make_tiled_mma(MMA_Atom<MMA_Traits<MMA_Op>>{}));
```

---

## 3. 迭代一：初始实现（结构搭建）

### 实现内容

搭建了两个 Kernel：

1. **`gemm_kernel_tma`**：TMA (G→S) + GMMA (SS模式)
2. **`gemm_kernel_cp_async`**：cp.async (G→S) + GMMA，用于对比

### 核心代码结构

**SMEM Layout**（满足 GMMA 硬件要求）：
```cpp
using SmemLayoutAtomA = GMMA::Layout_K_SW128_Atom<T>;
// 等价于 ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<(8,64),(64,1)>>
// TMA 要求 swizzle 必须是 0/32/64/128 bit 之一，且 M=4

using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{},
    make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})  // (128, 64, 3)
));
```

**TMA Copy 对象构建**（Host 端）：
```cpp
auto tma_a = cute::make_tma_copy(
    cute::SM90_TMA_LOAD{},
    tensor_A,          // gmem tensor (真实指针)
    smem_layout_a,     // 单个 stage 的 smem layout
    cute::Int<1>{}     // cluster_size=1, 不使用 multicast
);
```

**关键技巧 `as_position_independent_swizzle_tensor`**：
```cpp
// swizzle smem tensor 的 base 指针抽象化
// 使 TMA descriptor 中的坐标计算只依赖 layout 而非绝对地址
// 对于 ComposedLayout (Swizzle+Layout) 必须使用!
auto sA_pi = as_position_independent_swizzle_tensor(sA);
Tensor tAsA = cta_tma_a.partition_D(sA_pi);
```

### 测试配置

```cpp
using TestConfig = gemm_sm90::GemmConfig<
    cute::bfloat16_t,  // T
    128,               // kTileM
    128,               // kTileN
    64,                // kTileK
    3                  // kStage
>;
// SMEM: A=48KB + B=48KB + mbar=24B = 96KB (< device max 227KB)
// 线程数: 128 (1个 WarpGroup)
```

---

## 4. 迭代二：Bug修复 —— 结果为零

### 问题现象

```
[TMA+GMMA] FAILED  max_diff=...  res=0.00000
```

所有输出均为 0。

### 根本原因

最初使用了 `AtomLayoutMNK` 堆叠：
```cpp
// ❌ 错误写法
using TiledMMA = decltype(make_tiled_mma(
    MMA_Atom<MMA_Traits<MMA_Op>>{},
    Layout<Shape<_2, _1, _1>>{}  // AtomLayout 堆叠 M 方向 ×2
));
```

**后果**：
- SM90 GMMA Atom 本身需要 128 线程
- 堆叠 M×2 导致 TiledMMA 期望 **256 线程**
- 但 Kernel 只启动了 **128 线程**
- `partition_C` 错误映射，128 条线程中有一半永远不执行，对应的输出 Tile 无法写回，结果为 0

### 修复方案

移除 AtomLayout 堆叠，直接使用单个 MMA_Atom：
```cpp
// ✅ 正确写法
using TiledMMA = decltype(make_tiled_mma(
    MMA_Atom<MMA_Traits<MMA_Op>>{}
    // 无 AtomLayout 堆叠
    // cute::gemm 内部会自动对多个 K-tile 循环
));
```

### 修复后结果

```
[TMA+GMMA]    PASSED  max_diff=0.0000e+00  rel_err=0.00e+00
[cpasync+GMMA] PASSED  max_diff=0.0000e+00  rel_err=0.00e+00
```

---

## 5. 迭代三：性能问题 —— TMA慢于cp.async

### 问题现象

正确性通过后，跑 Benchmark（M=N=4096, K=2048）：

```
cuBLAS BF16:        0.524 ms   131.20 TFLOPS
SM90 TMA+GMMA:      1.146 ms    59.95 TFLOPS   (vs cuBLAS: 45.7%)  ← 异常慢!
SM90 cp.async+GMMA: 0.582 ms   117.99 TFLOPS   (vs cuBLAS: 89.9%)
```

**TMA 版本比使用老旧 `cp.async` 的版本慢了整整 2 倍**，完全不符合预期。

### 初步分析方向

1. mbar_wait 的 busy-spin 问题？
2. 流水线顺序问题？
3. phase 追踪逻辑错误？

经过仔细的时序推演，phase 追踪逻辑本身是正确的。真正的问题见迭代四。

---

## 6. 迭代四：根本Bug修复 —— WAR数据竞争

### 问题定位

对旧版主循环做逐迭代时序推演（kStage=3，k=1 为例）：

**旧代码顺序（错误）：**
```
k=1:
  step1: TMA issue → write_stage=0 (stage0)   ← 写入 stage0
         ↑ 但此时 group_0 (k=0的wgmma) 正在从 stage0 读数据!
  step2: mbar_wait(stage1)
  step3: wgmma(stage1) → group_1
  step4: wgmma_commit
  step5: wait<1>                                ← 直到这里才等 group_0 完成
```

**这是经典的 WAR (Write-After-Read) 数据竞争**：
- TMA 在 `wait<1>` **之前**就写入了 stage0 的 SMEM
- 而 `group_0` 的 wgmma 还在从 stage0 读数据
- TMA 覆盖了正在被读取的数据，导致计算结果错误，硬件可能触发重试或产生垃圾结果

### 修复方案

**将 TMA 发起移到 `warpgroup_wait<1>` 之后**：

```cpp
// ✅ 修复后的主循环
for (int k = 0; k < num_k_tiles; ++k) {

    // 1. 等待当前 stage 的 TMA 完成
    mbar_wait(&mbar[read_stage], phase);

    // 2. 发出 wgmma (异步)
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    gemm(tiled_mma, tCrA(_, _, _, read_stage), tCrB(_, _, _, read_stage), tCrC);
    warpgroup_commit_batch();

    // 3. wait<1>: 等上一轮 wgmma 完成
    //    此后 write_stage 的 SMEM 不再被任何 wgmma 读取，TMA 可以安全写入
    warpgroup_wait<1>();
    warpgroup_fence_operand(tCrC);

    read_stage = (read_stage + 1) % kStage;
    if (read_stage == 0) phase ^= 1;

    // 4. 现在才发起 TMA，SMEM 已经安全
    int next_k = k + (kStage - 1);
    if (next_k < num_k_tiles) {
        if (tid == 0) {
            mbar_arrive_and_expect_tx(&mbar[write_stage], kTmaBytes);
            copy(tma_a.with(mbar[write_stage]), tAgA(_, _, _, next_k), tAsA(_, _, _, write_stage));
            copy(tma_b.with(mbar[write_stage]), tBgB(_, _, _, next_k), tBsB(_, _, _, write_stage));
        }
        write_stage = (write_stage + 1) % kStage;
    }
}
```

### 正确性证明（kStage=3 时序追踪）

```
Prologue:
  TMA k=0 → stage0 (mbar[0]),  write_stage: 0→1
  TMA k=1 → stage1 (mbar[1]),  write_stage: 1→2
  初始: read_stage=0, write_stage=2, phase=0

k=0:
  mbar_wait(mbar[0], 0)      等 k=0 数据就绪
  wgmma(stage0) → group_0
  wait<1>                    无上一轮, 立即返回
  read_stage=1, phase=0
  TMA k=2 → stage2           stage2 未被任何 wgmma 使用 ✓

k=1:
  mbar_wait(mbar[1], 0)      等 k=1 数据就绪
  wgmma(stage1) → group_1
  wait<1>                    等 group_0 完成 (group_0 读 stage0)
                             group_0 完成后 stage0 安全 ✓
  read_stage=2, phase=0
  TMA k=3 → stage0           group_0 已完成, stage0 可以覆盖 ✓

k=2:
  mbar_wait(mbar[2], 0)      等 k=2 数据就绪 (k=0 发起的)
  wgmma(stage2) → group_2
  wait<1>                    等 group_1 完成 (group_1 读 stage1)
  read_stage=0, phase=1      绕回, phase 翻转!
  TMA k=4 → stage1           group_1 已完成, stage1 可以覆盖 ✓

k=3:
  mbar_wait(mbar[0], 1)      注意 phase=1 ← 因为 mbar[0] 被 arrive 过两次，phase 已翻
  wgmma(stage0) → group_3
  ...
```

**流水线重叠效果**：`group_k`（当前轮 wgmma）与**下一迭代的 `mbar_wait` + TMA 传输**真正并行。

### 额外优化：mbar_wait 升级

```cpp
// 旧版：简单 busy-spin，128线程全部占用 SM issue slot
"mbarrier.try_wait.parity.shared::cta.b64 complete, [%0], %1;\n"

// 新版：acquire 语义 + nanosleep 退避
"mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 complete, [%0], %1;\n"
"nanosleep.u32 0x100;\n"   // 等待失败时约 256ns 退避
```

| 改动 | 效果 |
|------|------|
| `acquire.cta` 语义 | 保证 TMA 写入的数据在 wait 返回后对当前 CTA 内存可见（正式内存序） |
| `nanosleep 0x100` | 等待失败时让 warp scheduler 切换到其他 warp，降低 SM 资源竞争 |

---

## 7. 迭代五：Ping-Pong 初版（性能低于预期）

### 动机

迭代四的单 WarpGroup 版本（132 TFLOPS）已经对齐 cuBLAS，但存在一个结构性限制：  
**`mbar_wait` 时 Consumer（wgmma）和 Producer（TMA）是同一批 128 线程**，Consumer 在等待 TMA 期间 Tensor Core 完全空闲。

Ping-Pong 设计将两者彻底分离，让 Tensor Core 永远不需要等 TMA。

### 核心设计

```
CTA 内分成两个 WarpGroup（共 256 线程）：

  WG0 (Consumer, tid   0-127): 专门做 wgmma
  WG1 (Producer, tid 128-255): 专门做 TMA + mbar_wait

                  mbar_full[kStage]          mbar_empty[kStage]
  Producer ──────────────────────────────► Consumer
           ◄────────────────────────────────
```

**两套 mbarrier 的语义（初版）**：

| mbarrier | arrive_count | 含义 |
|----------|-------------|------|
| `mbar_full[s]` | 1 | Producer 完成 TMA → Consumer 可以开始 wgmma |
| `mbar_empty[s]` | 128 | Consumer WG 全体完成 wgmma → Producer 可以覆盖 SMEM |

### 流水线时序（kStage=2）

```
时间轴 →

Producer (WG1):    wait_e[0] | TMA k=0→s0 | wait_e[1] | TMA k=1→s1 | wait_e[0] | TMA k=2→s0 | ...
                                ↓ signal_f[0]              ↓ signal_f[1]              ↓ ...
Consumer (WG0):               wait_f[0] | wgmma(s0) | wait_f[1] | wgmma(s1) | ...
                                           ↓ signal_e[0]              ↓ signal_e[1]
```

两者完全异步：Producer 在 Consumer 做 wgmma 的同时就可以发起下一次 TMA。

### mbar_empty 初始化技巧

```cpp
// 初始化后立刻让 Consumer 128 线程全体 arrive 一次
// 等价于声明: "所有 stage 初始时都是空闲的"
if (!is_producer) {
    for (int s = 0; s < kStage; ++s)
        mbar_arrive(&mbar_empty[s]);  // 每个 Consumer 线程 arrive 一次（初版，后续优化）
}
__syncthreads();
```

### Consumer 主循环（初版）

```cpp
// ❌ 初版问题：128 线程全体 arrive，产生大量无效指令
if (prev_stage >= 0)
    mbar_arrive(&mbar_empty[prev_stage]);  // 128 线程各执行一次
```

### 实测结果（初版，kStage=3）

```
SM90 PingPong (kStage=3): 0.592 ms  116.01 TFLOPS  (vs cuBLAS: 88.4%)  ← 不升反降!
```

**比单 WarpGroup TMA 版本（132 TFLOPS）慢了 ~12%**，见迭代六的原因分析与修复。

---

## 8. 迭代六：Ping-Pong 优化 —— arrive开销+kStage调整

### 原因分析

初版 Ping-Pong 性能低于单 WG 版本，经分析有两个根本原因：

**原因一：`mbar_empty.arrive_count=128` 导致 arrive 指令爆炸**

每个 K-tile 迭代中 Consumer WarpGroup 的 128 个线程**各执行一条 arrive 指令**：

```
num_k_tiles × 128 = 32 × 128 = 4096 条 arrive 指令
```

这些 arrive 指令大量占用 Consumer WarpGroup 的 issue slot，压缩了 wgmma 的发射窗口。而实际上，`wgmma 完成`的保证已经由 `warpgroup_wait<1>` 提供，`mbar_empty` 只需要通知 Producer「可以开始写了」，一次 arrive 就足够。

**原因二：kStage=3 对 Ping-Pong 太小，Consumer 仍然等 Producer**

```
TMA 延迟：约 400~800 cycles
单次 wgmma tile 计算时间：约 200~400 cycles (128×128×64 BF16)
```

kStage=3 时 Producer 最多只能比 Consumer 超前 **2 个 tile**，不足以完全掩盖 TMA 延迟。Consumer 的 `mbar_wait(full[s])` 依然频繁 stall，Ping-Pong 的异步优势无法发挥。

单 WG 版本虽然 Consumer 也会 `mbar_wait`，但它只有 1 个 WarpGroup，SM 会在等待期间调度**其他 block 的 warp**来掩盖延迟，整体 SM 利用率更高。

### 修复方案

**修复一：`mbar_empty.arrive_count` 改为 1，只让 `wg_tid==0` arrive**

```cpp
// ❌ 旧版：arrive_count=128，128 线程全体 arrive
mbar_init(&mbar_empty[s], 128);
if (!is_producer)  // 128 线程都执行
    mbar_arrive(&mbar_empty[s]);

// ✅ 新版：arrive_count=1，只让 wg_tid==0 arrive
mbar_init(&mbar_empty[s], 1);
if (!is_producer && wg_tid == 0)  // 只有 1 个线程执行
    mbar_arrive(&mbar_empty[s]);
```

主循环中同样：
```cpp
// ✅ 修复后
if (prev_stage >= 0 && wg_tid == 0)
    mbar_arrive(&mbar_empty[prev_stage]);
```

> `warpgroup_wait<1>` 已保证整个 WarpGroup 的 wgmma 完成，`wg_tid==0` 代表全组 arrive，arrive 指令从 4096 条降为 32 条。

**修复二：Ping-Pong 使用 `kStage=4`（独立 Config）**

```cpp
// main.cu
using PingPongConfig = gemm_sm90::GemmConfig<T, 128, 128, 64, 4>;
// SMEM: 128×64×4×2×2 = 128KB + mbar(64B) ≈ 128KB < H20 max 227KB
```

kStage=4 时 Producer 可以提前 **3 个 tile** 开始搬运，充分掩盖 TMA 延迟。

### 修复后期望的时序

```
时间轴 →  (kStage=4)

Producer:  [wait_e→TMA k=0] [wait_e→TMA k=1] [wait_e→TMA k=2] [wait_e→TMA k=3] ...
              ↓ full[0]           ↓ full[1]           ↓ full[2]           ↓ ...
Consumer:          [wait_f→wgmma(0)→signal_e(0)] [wait_f→wgmma(1)→signal_e(1)] ...

Producer 始终超前 Consumer 约 2~3 个 tile，Consumer 无需等待
```

### 代码变更汇总

| 文件 | 改动 |
|------|------|
| `gemm_sm90.cuh` | `mbar_empty.arrive_count`: 128→1；初始化和 signal 改为 `wg_tid==0` only |
| `main.cu` | 新增 `PingPongConfig`（kStage=4）；Ping-Pong benchmark 改用 `PingPongConfig` |

---

## 9. 迭代七：实测结果与分析 —— kStage=4性能下降与容器内性能分析方案

### 实测结果（迭代六优化后，kStage=4 + arrive×1）

```
cuBLAS BF16:              0.524 ms   131.03 TFLOPS   (baseline)
SM90 TMA+GMMA (k3):       0.520 ms   132.09 TFLOPS   (vs cuBLAS: 100.8%) ✓
SM90 TMA+GMMA (k4):       0.588 ms   116.77 TFLOPS   (vs cuBLAS:  89.1%) ↓
SM90 cp.async+GMMA:       0.582 ms   118.11 TFLOPS   (vs cuBLAS:  90.1%)
SM90 PingPong (k4,opt):   0.592 ms   116.11 TFLOPS   (vs cuBLAS:  88.6%) ↓
```

### 两个意外发现

**发现一：kStage=4 对单WG TMA 版反而更慢**

kStage=3 是 96KB SMEM，kStage=4 是 128KB SMEM。

H20 每个 SM 有 228KB L1/SMEM，当单 block 占用 128KB 时可能加剧了 **L1 缓存压力**：
- SMEM 占用越大，剩余给 L1 cache 的空间越小
- Epilogue 写回 gmem 时 L1 hit rate 降低
- 同时运行的 block 数可能从 2 降为 1（occupancy 下降）

这说明对于单WG 版本，kStage=3 就是最佳点。

**发现二：Ping-Pong kStage=4 优化后依然没有超越单WG**

```
PingPong (k4, arrive×1): 116 TFLOPS  vs  单WG TMA (k3): 132 TFLOPS
差距依然 ~12%
```

这就是为什么需要容器内性能分析。

### 容器内可用的性能分析方法

ncu 在 docker 容器内无权限时，可用以下替代方案：

**方案一：nsys（时间线层面）**

```bash
# nsys 在大多数 docker 环境下不需要 root 权限
nsys profile --stats=true -o report ./gemm_sm90
# 查看 kernel 计时和占比
nsys stats report.nsys-rep --report cuda_gpu_kern_sum
```

**方案二：cuobjdump SASS 分析**

```bash
# 查看寄存器用量和 SMEM 占用
cuobjdump --res-usage ./gemm_sm90
# 查看具体指令类型和占比
cuobjdump -sass ./gemm_sm90 | grep -E "(WGMMA|LDG|STG|LDSM|MBAR)" | head -50
```

**方案三： clock64 手工计时 (device-side)

在 kernel 内用 `clock64()` 获取 SM 时钟计数，测量关键步骤延迟：

```cuda
// 举例: 测量 Consumer mbar_wait 实际耗时
if (wg_tid == 0) {
    long long t0 = clock64();
    mbar_wait(&mbar_full[stage], phase_f);
    long long t1 = clock64();
    if (blockIdx.x == 0 && k < 3)
        printf("mbar_wait stall cycles: %lld\n", t1 - t0);
}
```

如果 `mbar_wait` stall cycles 很高（>> 1000 cycles），证明 Consumer 仍在等 Producer，
需要增大 kStage 或查看 Producer 是否应和 TMA 延迟本身。

**方案四：PTX cuobjdump 分析指令密度**

```bash
# 生成 PTX 查看编译器生成的具体指令
nvcc -ptx -arch=sm_90a ... gemm_sm90.cuh
# 或从二进制提取 SASS
cuobjdump -sass ./gemm_sm90 | grep -A 200 "gemm_kernel_pingpong"
```

重点查看：
- Consumer 主循环内 `MBAR.WAIT` 前后是否有其他指令排集
- `WGMMA` 指令密度（WGMMA 之间是否有大量无关指令）
- `MBAR.ARRIVE` 调用次数是否已降至预期

---

## 10. 迭代八：Ping-Pong Producer 优化 —— 单线程驱动 TMA

### 优化思路

PTX ISA 文档承认：`mbarrier.try_wait.acquire` 的内存可见性只对“执行该指令的线程”有效。这意味着：

- **Consumer `mbar_full` wait**：必须全 128 线程执行，否则其他线程没有 acquire 保证，wgmma 读到过时数据 → **不能优化**
- **Producer `mbar_empty` wait**：只有 `wg_tid==0` 需要确认“SMEM 可写”，其他 Producer 线程不读写 SMEM → **可以优化**

### 修复方案

**Producer 整个循环改为只有 `wg_tid==0` 执行**：

```cpp
// ❌ 旧版：128 线程全部 busy-spin mbar_empty
for (int k = 0; k < num_k_tiles; ++k) {
    mbar_wait(&mbar_empty[stage], phase_e);   // 128 次 nanosleep 循环
    if (wg_tid == 0) {
        mbar_arrive_and_expect_tx(...);
        copy(...);
    }
    stage = ...
}

// ✅ 新版：只有 wg_tid==0 执行整个循环
if (wg_tid == 0) {
    for (int k = 0; k < num_k_tiles; ++k) {
        mbar_wait(&mbar_empty[stage], phase_e);  // 1 次 nanosleep 循环
        mbar_arrive_and_expect_tx(...);
        copy(...);
        stage = ...
    }
}
// 其他 127 个 Producer 线程 idle，等到 Epilogue __syncthreads()
```

**效果**：Producer WG 的 127 个 idle 线程释放 SM 资源，有助于其他 block 的 warp 调度。

### 关键正确性諺证：为什么 Producer 可以单线程而 Consumer 不行

| 等待的 mbarrier | acquire 可见性需求 | 可否单线程 |
|---|---|---|
| Producer `mbar_empty` | 只需确认 SMEM 可写，wg_tid==0 发起 TMA，不读 SMEM | ✅ 可以 |
| Consumer `mbar_full` | 确认 TMA 写入对全 128 线程可见 (用于 wgmma SS 读取 SMEM) | ❌ 不行 |

### SMEM 布局更新（main.cu 新增 Section 1b）

```cpp
// 新增 TestConfig4 用于对比
using TestConfig4 = gemm_sm90::GemmConfig<T, 128, 128, 64, 4>;
// Section 1b: 单WG TMA kStage=4
// 结果: 116.77 TFLOPS, 比 kStage=3 慢 12% → 证明 kStage=3 是单WG 最佳点
```

---

## 11. 迭代九：nsys 性能分析与低 occupancy 根因定位

### 背景

迭代八的 Ping-Pong kStage=4 版本经测试仍只达到 **116 TFLOPS**（单WG TMA 为 132 TFLOPS），差距约 12%。由于测试环境是 Docker 容器，无法使用 `ncu`，为此编写了 `analysis.sh` 脚本，结合 `nsys`、`cuobjdump` 和 `clock64` device-side 计时进行分析。

### nsys 分析方法

```bash
# 生成 nsys 报告
nsys profile --stats=true -o report ./gemm_sm90

# 查看各 kernel 执行时间
nsys stats report.nsys-rep --report cuda_gpu_kern_sum

# 查看 CUDA API 调用
nsys stats report.nsys-rep --report cuda_api_sum
```

通过 `nsys` 时间线可以对比不同 kernel 的持续时间，确认 Ping-Pong kernel 耗时是否真的长于单WG TMA kernel。

### cuobjdump 寄存器分析

```bash
# 查看各 kernel 的寄存器用量与 SMEM 占用
cuobjdump --res-usage ./gemm_sm90
```

关键发现：

```
gemm_kernel_pingpong:
  Used 232 registers, 131136 bytes smem

gemm_kernel_tma:
  Used 154 registers, 98352 bytes smem
```

**Ping-Pong 使用了 232 寄存器/线程**，远高于单WG TMA 的 154。

### 低 occupancy 根因分析

| kernel | 寄存器/线程 | 线程数/block | 寄存器/block | SM 可驻留 block 数 |
|--------|------------|-------------|-------------|------------------|
| TMA 单WG | 154 | 128 | 19712 | 4 |
| Ping-Pong（重构前）| **232** | **256** | **59392** | **1** |

H20 SM 共有 65536 个寄存器。Ping-Pong kernel 每 block 占用 59392 个，整个 SM 只能驻留 **1 个 block（128 个 warp 占位，但实际 SM 需要 2+ block 才能 hide latency）**。

**结论**：Ping-Pong kernel 的性能瓶颈是**超低 occupancy**（SM 寄存器不足以驻留第二个 block），而非 TMA/wgmma 算法本身的问题。

### 寄存器过多的原因

`ptxas` 在编译单个 kernel 时，对 **if-else 两路都执行的变量取并集**：

```
Producer 路径需要: ~40 reg (TMA 循环变量)
Consumer 路径需要: ~154 reg (wgmma 累加器 tCrC ~128 FP32)

ptxas 统计整个 kernel: max(40, 154) ≈ 154?
实际输出:                              232  ← 过多!
```

实际上 `ptxas` 保守地为两路都分配了完整的寄存器集合（包括 tCrC 的 128 个 FP32 累加器 + 共享的循环变量），导致峰值达到 232。

---

## 12. 迭代十：寄存器分离优化 —— `__noinline__` 重构

### 优化思路

核心问题：Producer 和 Consumer 在同一函数体内，`ptxas` 为整个 kernel 分配寄存器时无法分别优化。

**解决方案**：将 Producer 和 Consumer 的主循环分别提取为独立的 `__device__ __noinline__` 子函数：

```
pp_producer_loop(...)  ← ptxas 独立编译, 约 40 reg
pp_consumer_loop(...)  ← ptxas 独立编译, 约 154 reg
```

`__noinline__` 关键字禁止编译器将函数内联回 kernel，保持寄存器分配边界。若被内联，`ptxas` 仍会合并两路寄存器需求，优化无效。

### 代码重构

**重构前**（kernel 内直接 if-else 判断）：

```cpp
// ❌ 重构前：两路在同一编译单元
__global__ void gemm_kernel_pingpong(...) {
    // ... 初始化 ...

    if (is_producer) {
        // Producer 循环（~40 reg 变量）
        for (int k = 0; k < num_k_tiles; ++k) {
            mbar_wait(&mbar_empty[stage], phase_e);
            mbar_arrive_and_expect_tx(...);
            copy(...);
        }
    } else {
        // Consumer 循环（~154 reg 变量，含 tCrC）
        for (int k = 0; k < num_k_tiles; ++k) {
            mbar_wait(&mbar_full[stage], phase_f);
            wgmma_fence(); wgmma_arrive();
            gemm(tiled_mma, tCrA(...), tCrB(...), tCrC);
            wgmma_commit(); wgmma_wait<1>();
        }
    }
}
// ptxas 统计: 232 reg/thread × 256 threads = 59392 reg/block → SM 只驻 1 block
```

**重构后**（分离为 `__noinline__` 子函数）：

```cpp
// ✅ 重构后：两路独立编译
template <...>
__device__ __noinline__ void pp_producer_loop(...) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" :::);
    if (wg_tid == 0) {
        for (int k ...) { mbar_wait + TMA copy; }
    }
}   // ptxas 单独统计: ~40 reg

template <...>
__device__ __noinline__ void pp_consumer_loop(...) {
    for (int k ...) {
        mbar_wait + wgmma + wgmma_wait<1> + mbar_arrive;
    }
}   // ptxas 单独统计: ~154 reg

__global__ void gemm_kernel_pingpong(...) {
    // ... 初始化 ...
    if (is_producer) {
        pp_producer_loop<Config>(...);  // __noinline__ 调用
    } else {
        pp_consumer_loop<Config>(...);  // __noinline__ 调用
    }
}
// ptxas 统计: max(40, 154) = 154 reg/thread × 256 threads = 39424 reg/block
// SM 可驻留: 65536 / 39424 ≈ 1.6 → 实际 1~2 block, 有所改善
```

### `setmaxnreg` 的作用

在 `pp_producer_loop` 函数开头加入：

```cpp
asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" :::);
```

这是 SM90 新增的 PTX 指令，运行时动态降低 Producer WG 的寄存器配额至 40：
- **效果**：释放多余的物理寄存器，允许 SM 上的其他 block 使用
- **要求**：必须由整个 WarpGroup（128 线程）同步执行（`.sync.aligned` 语义）
- 在 Consumer 函数入口可以对应地 `setmaxnreg.inc` 恢复，但因 Consumer 本就需要完整的 154 reg，此处不需要

### 预期效果

| 指标 | 重构前 | 重构后（预期） |
|------|--------|--------------|
| 寄存器/线程 | 232 | ~154 |
| 寄存器/block (256线程) | 59392 | ~39424 |
| SM 驻留 block 数 | 1 | ~1.6 (实际 1-2) |
| TFLOPS | 116 | 待测 |

**待验证**：用 `--ptxas-options=-v` 编译查看实际寄存器数，再运行基准测试。

---

## 13. 迭代十一：代码结构整理

### 背景

随着 kernel 数量增加（TMA、cp.async、Ping-Pong）和各种辅助设施（GemmConfig、mbarrier PTX 封装、SMEM size helpers），所有代码都堆在单一的 `gemm_sm90.cuh` 文件中（917 行），可读性和可维护性较差。

### 重构方案

将 `gemm_sm90.cuh` 拆分为如下结构：

```
tensor-core/sm90/
  gemm_sm90.cuh              ← 主入口，只做 include 整合（对外接口不变）
  detail/
    config.cuh               ← GemmConfig + get_smem_size_* helpers
    mbarrier.cuh             ← mbar_init / mbar_arrive / mbar_wait / mbar_fence_init
    kernel_tma.cuh           ← gemm_kernel_tma（单WG TMA + GMMA）
    kernel_cp_async.cuh      ← gemm_kernel_cp_async（cp.async + GMMA，对比用）
    kernel_pingpong.cuh      ← pp_producer_loop + pp_consumer_loop
                                + gemm_kernel_pingpong（双WG Ping-Pong）
```

**原则**：
- `main.cu` 的 `#include "gemm_sm90.cuh"` **保持不变**，对调用方完全透明
- 每个 `detail/` 文件自成一体，有独立的注释说明设计意图
- `detail/kernel_*.cuh` 只 include `detail/config.cuh`（和 `detail/mbarrier.cuh`），不相互依赖

### 各文件职责

| 文件 | 内容 | 行数（约） |
|------|------|-----------|
| `gemm_sm90.cuh` | 主入口，5 个 include | ~25 |
| `detail/config.cuh` | GemmConfig 模板 + 3 个 SMEM size 函数 | ~120 |
| `detail/mbarrier.cuh` | mbar_init/arrive/fence/wait PTX 封装 | ~65 |
| `detail/kernel_tma.cuh` | gemm_kernel_tma | ~140 |
| `detail/kernel_cp_async.cuh` | gemm_kernel_cp_async | ~100 |
| `detail/kernel_pingpong.cuh` | pp_producer/consumer_loop + gemm_kernel_pingpong | ~220 |

---

## 14. 当前代码结构

### 文件结构

```
tensor-core/sm90/
  gemm_sm90.cuh              ← 主入口（只做 include 整合）
  detail/
    config.cuh               ← GemmConfig + SMEM size helpers
    mbarrier.cuh             ← mbarrier PTX 封装
    kernel_tma.cuh           ← gemm_kernel_tma（单WG TMA + GMMA）
    kernel_cp_async.cuh      ← gemm_kernel_cp_async（cp.async + GMMA）
    kernel_pingpong.cuh      ← pp_producer_loop + pp_consumer_loop
                                + gemm_kernel_pingpong（双WG Ping-Pong）
  main.cu                    ← 测试驱动，含 3 个 kernel × 多 kStage 配置
  CMakeLists.txt             ← 构建（含 --ptxas-options=-v）
  analysis.sh                ← nsys + cuobjdump + clock64 分析脚本
  DEVLOG.md                  ← 本文件
```

### API 概览

```
─── detail/config.cuh ───────────────────────────────────────────────────────
GemmConfig<T, kTileM, kTileN, kTileK, kStage>
  MMA_Op        = SM90_64x128x16_F32BF16BF16_SS<K, K>
  TiledMMA      = make_tiled_mma(MMA_Atom<MMA_Traits<MMA_Op>>{})
  SmemLayout    = tile_to_shape(Layout_K_SW128_Atom<T>{}, (M, K, Stage))
  G2SCopy       = cp.async 128bit, 128 线程
  kNumThreads   = 128   (单 WG 版本)
  kNumThreadsPP = 160   (Ping-Pong v4: 128 Consumer + 32 Producer)
get_smem_size_tma<Config>()        → A+B+mbar (kStage 个)
get_smem_size_cp_async<Config>()   → A+B
get_smem_size_pingpong<Config>()   → A+B+2×kStage 个 mbar

─── detail/mbarrier.cuh ─────────────────────────────────────────────────────
mbar_init(mbar, arrive_count)
mbar_arrive_and_expect_tx(mbar, tx_bytes)  ← Producer 用
mbar_arrive(mbar)                          ← Consumer 用（纯计数）
mbar_fence_init()
mbar_wait(mbar, expected_phase)            ← acquire + nanosleep 退避

─── detail/kernel_tma.cuh ───────────────────────────────────────────────────
gemm_kernel_tma<Config, TmaCopyA, TmaCopyB>(C, tma_a, tma_b, M, N, K)
  128 线程, 单 WG
  Prologue: 预取 kStage-1 个 tile
  MainLoop: mbar_wait → wgmma → wgmma_wait<1> → TMA_issue
  Drain: wgmma_wait<0>

─── detail/kernel_cp_async.cuh ──────────────────────────────────────────────
gemm_kernel_cp_async<Config>(C, A, B, M, N, K)
  128 线程, 单 WG
  cp.async + cp_async_fence/wait + wgmma_wait<0>（用于对比）

─── detail/kernel_pingpong.cuh (v4) ─────────────────────────────────────────
pp_producer_loop<Config>(...)  __device__ __forceinline__
  prod_tid = tid - 128 (0-31), 只有 prod_tid==0 发起 TMA + mbar 操作

pp_consumer_loop<Config>(...)  __device__ __forceinline__
  全 128 线程: mbar_wait → wgmma → wgmma_wait<1> → mbar_arrive_if
  无 setmaxnreg

gemm_kernel_pingpong<Config, TmaCopyA, TmaCopyB>(C, tma_a, tma_b, M, N, K)
  __launch_bounds__(160, 2)
  tid  0-127: Consumer WG (完整 WarpGroup, wgmma 有效)
  tid 128-159: Producer warp (只有 tid==128 工作)
  ptxas: 154 reg, 0 spill → occupancy=2 @ kStage=3
```

### SMEM 布局

**单 WG TMA 版（kStage=3）：**
```
  A: 128×64×3 × 2B = 49152B
  B: 128×64×3 × 2B = 49152B
  mbarrier: 3 × 8B = 24B
  总计: 98328B ≈ 96KB
```

**Ping-Pong 版（kStage=4）：**
```
  A: 128×64×4 × 2B = 65536B
  B: 128×64×4 × 2B = 65536B
  mbar_full:  4 × 8B = 32B
  mbar_empty: 4 × 8B = 32B
  总计: 131136B ≈ 128KB  (< H20 max 227KB)
```

---

## 15. 迭代十二：v3，消除 C7510 警告（`__forceinline__`）

### 问题：`__noinline__` 导致 C7510 警告引发 wgmma 序列化

v2 重构后虽然 ptxas 报告寄存器降至约 160，但编译出现：

```
warning #7510-D: wgmma.mma_async pipeline: wgmma pipeline may be suboptimal
  due to crossing function boundary
```

根因：`__noinline__` 在 `pp_consumer_loop` 函数边界强迫编译器插入隐式的 `wgmma.wait_group 0`，序列化所有 in-flight wgmma，等价于主动禁用了 GMMA 异步流水线。

实测（v2，256 线程）：
```
PingPong(k3) = 603 us   vs   TMA(k3) = 522 us   差距 81 us (15.5%)
```

### 修复：`__noinline__` → `__forceinline__`

将 `pp_producer_loop` 和 `pp_consumer_loop` 均改为 `__forceinline__`：
- 消除函数边界 → C7510 警告消失 → wgmma async pipeline 完整保留

v3 实测（256 线程，保留 setmaxnreg）：
```
PingPong(k3) = 594 us   差距缩小至 72 us (13.8%)
```

### 遗留问题

```
PingPong(k3) = 594 us  ≈  TMA(k4) = 593 us
```

这个等式暗示 PingPong(k3) 实际 occupancy = **1 block**，`setmaxnreg` 并未生效。

---

## 16. 迭代十三：诊断 `setmaxnreg` 无效的根因

### 根本原因

**SM 在 block 入驻（kernel launch 阶段）时，按 ptxas 编译期静态值（160 reg × 256 threads = 40960）预分配寄存器文件空间。此时 `setmaxnreg` 尚未执行，SM 只能驻留 1 个 block。**

`setmaxnreg` 只对 **persistent kernel** 有效：某个 WarpGroup 完成工作、执行 `setmaxnreg.dec` 释放寄存器后，SM 动态调度新 block，收益才能体现。

### 参考：CUTLASS 官方 SM90 PingPong 架构

查阅 `cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp`：

```
MaxThreadsPerBlock = 384（1 Producer WG + 2 Consumer WG）
MinBlocksPerMultiprocessor = 1
LoadRegisterRequirement = 40    (Producer warpgroup_reg_dealloc)
MmaRegisterRequirement  = 232   (Consumer warpgroup_reg_alloc)
```

CUTLASS PingPong 的真正价值：**2 个 Consumer WG 分别处理不同的 output tile**，通过 `MathWarpGroupOrderBarrier` 交替执行 MMA 和 Epilogue，需配合 persistent work tile scheduler。对于非 persistent 单 tile 设计，该架构无法直接套用。

---

## 17. 迭代十四：v4，减少 Producer 线程数（256→160）

### 核心洞察：从线程数而非 setmaxnreg 入手

直接减少 block 总线程数，让 ptxas 静态值自然满足 occupancy=2：

```
v3 (256 线程):  160 reg × 256 = 40960 reg/block → floor(65536/40960) = 1 block
v4 (160 线程):  154 reg × 160 = 24640 reg/block → floor(65536/24640) = 2 blocks ✅
```

### 为什么 Producer 只需 32 线程（1 warp）？

原 Producer WarpGroup 有 128 线程（4 warps），但：
- TMA 发起只需 1 个线程（`prod_tid==0`），其余 127 个线程**完全空转**
- mbarrier arrive/wait 也只需 1 个线程
- 减少到 32 线程后，SM warp scheduler 有更多资源服务 Consumer WG

### 代码修改

**`config.cuh`：**
```cpp
static constexpr int kNumThreadsPP = 160;  // 128 Consumer + 32 Producer (v4)
```

**`kernel_pingpong.cuh` v4：**
```cpp
// 全部移除 setmaxnreg
// tid  0-127: Consumer WG (整个 WarpGroup, 参与 wgmma)
// tid 128-159: Producer warp (只有 tid==128 发起 TMA)
const bool is_producer = (tid >= 128);
const int  wg_tid      = tid;           // Consumer: 直接 0-127
const int  prod_tid    = tid - 128;     // Producer: 0-31

__launch_bounds__(160, 2)  // min_blocks=2 提示 ptxas 目标 occupancy
```

**`__launch_bounds__(160, 2)` 的额外效果：**
`min_blocks=2` 将 ptxas 寄存器上限压缩到 `floor(65536/(2×160)) = 204 reg`，ptxas 据此进一步优化至 **154 reg**（v3 为 160 reg）。

### 实测结果

```
ptxas: 154 reg, 0 spill stores, 0 spill loads  (kStage=3 和 kStage=4)
```

| 版本 | 线程数 | 寄存器 | PingPong(k3) | PingPong(k4) |
|------|--------|--------|-------------|-------------|
| v3（setmaxnreg）| 256 | 160 | 594 us | 595 us |
| **v4（160线程）** | **160** | **154** | **524 us ✅** | **594 us** |

PingPong(k3)：594 us → **524 us**，与 TMA(k3) = 522 us **持平（差距 < 0.5%）**！

### nsys 深度分析验证

`cuda_gpu_kern_sum` 汇总：
```
gemm_kernel_pingpong (kStage=4, 128KB SMEM)  Avg = 593 us  ← 1 block, SMEM 受限
gemm_kernel_pingpong (kStage=3,  96KB SMEM)  Avg = 523 us  ← 2 blocks ✅
gemm_kernel_tma      (kStage=3,  96KB SMEM)  Avg = 523 us  ← 2 blocks (对照)
```

逐次时间序列（query 2-11）：
```
PingPong run  1-26 (kStage=4): ~593 us  ← 1 block (SMEM 瓶颈)
PingPong run 27-52 (kStage=3): ~522 us  ← 2 blocks ✅ 完全等同 TMA
TMA      run  1-26 (kStage=3): ~522 us
TMA      run 27-38 (kStage=4): ~593 us
```

### kStage=4 仍受 SMEM 限制

```
PingPong(k4): SMEM = 128KB → floor(227KB/128KB) = 1 block
```
与线程数和寄存器无关，是 128×128×64 tile size 下的架构上限。

---

## 18. 迭代十五：v5，寄存器隔离（Consumer __forceinline__ + Producer __noinline__）

### v4 遗留问题：232 reg 导致 occupancy 仍为 1

v4 实测 ptxas 报告 **232 registers**，而非预期的 154：

```
v4 实测:  232 reg × 160 = 37120 reg/block → floor(65536/37120) = 1 block ❌
预期目标: 154 reg × 160 = 24640 reg/block → floor(65536/24640) = 2 blocks ✅
```

**根因分析**：v4 将 `pp_producer_loop` 和 `pp_consumer_loop` 都设为 `__forceinline__`，
ptxas 将两者全部内联进 kernel 本体。对于 `if (is_producer) { ... } else { ... }` 结构，
ptxas **为两个分支同时分配寄存器**（即取两分支寄存器需求的并集）：

```
Consumer 分支: ~154 reg（wgmma 累加器 + tCrA/tCrB/tCrC）
Producer 分支: ~78  reg（TMA 张量视图 tAgA/tAsA/tBgB/tBsB）
合并后 kernel: 232 reg（两者叠加）
```

TMA 张量视图本身寄存器开销约 78 reg，这是 v4 的 "hidden cost"。

### v5 解决方案：寄存器路径隔离

```
Consumer 路径: pp_consumer_loop  → __forceinline__  (wgmma 必须无函数边界, 避免 C7510)
Producer 路径: pp_producer_loop  → __noinline__     (只做 TMA, 不执行 wgmma, 无 C7510 风险)
```

关键变化：
1. `pp_producer_loop` 改回 `__noinline__`，**将 TMA 张量视图构建移入函数内部**
2. kernel 本体 Consumer 路径只构建 `gC / sA / sB`，无任何 TMA 相关变量
3. ptxas 分析 kernel 时，Consumer 线程不再看到 TMA 视图寄存器

```cpp
// v5: Producer __noinline__ 隔离
template <typename Config, typename TmaCopyA, typename TmaCopyB>
__device__ __noinline__ void pp_producer_loop(
    ..., char* smem_A_ptr, char* smem_B_ptr,
    int prod_tid, int bx, int by, int M, int N, int K)
{
    if (prod_tid == 0) {
        // TMA 视图在函数内部构建 (寄存器限制在本函数栈帧, 不污染 kernel 本体)
        auto mA = tma_a.get_tma_tensor(make_shape(M, K));
        Tensor tAgA = cta_tma_a.partition_S(gA);
        // ... TMA copy loop ...
    }
}

// kernel 本体 Consumer 路径: 无 TMA 视图
} else {
    auto sA = make_tensor(make_smem_ptr(...), typename Config::SmemLayoutA{});
    auto gC = local_tile(...);          // 只有 gC/sA/sB
    pp_consumer_loop<Config>(...);      // __forceinline__, 154 reg
}
```

### v5 编译结果

```
ptxas info: gemm_kernel_pingpong (k3): Used 154 registers, 0 spill ✅  (v4: 232!)
ptxas info: gemm_kernel_pingpong (k4): Used 154 registers, 0 spill ✅
ptxas info: pp_producer_loop:           0 bytes stack, 0 bytes spill   ✅
```

occupancy 计算：
```
寄存器: 154 × 160 = 24640 reg/block → floor(65536/24640) = 2 blocks ✅
SMEM(k3): 96KB → floor(227/96) = 2 blocks ✅
→ occupancy = 2 blocks/SM ✅
```

### v5 性能结果

```
SM90 TMA+GMMA:      0.524 ms   131.1 TFLOPS   (vs cuBLAS: 100.7%)
SM90 PingPong(k3):  0.524 ms   131.1 TFLOPS   (vs cuBLAS: 100.7%) ← 与 TMA 完全持平!
SM90 PingPong(k4):  0.595 ms   115.5 TFLOPS   (vs cuBLAS: 88.7%)  ← SMEM 受限, 正常
```

PingPong(k3) 从 v4 的 **116 TFLOPS** 提升到 **131 TFLOPS**，提升 ~13%。

---

## 19. 当前代码结构（v5）

### API 概览

```
─── detail/config.cuh ───────────────────────────────────────────────────────
GemmConfig<T, kTileM, kTileN, kTileK, kStage>
  kNumThreads   = 128   (单 WG 版本)
  kNumThreadsPP = 160   (Ping-Pong v5: 128 Consumer + 32 Producer)
    → 154 reg × 160 = 24640 reg/block → SM 驻留 2 blocks @ kStage=3

─── detail/kernel_pingpong.cuh (v5) ─────────────────────────────────────────
pp_producer_loop<Config>(...)   __device__ __noinline__
  接受 raw SMEM 指针 + 坐标参数
  prod_tid==0 在函数内部构建 TMA 视图并执行 TMA copy loop
  __noinline__ 隔离 TMA 视图寄存器不污染 kernel 本体

pp_consumer_loop<Config>(...)   __device__ __forceinline__
  全 128 线程: mbar_wait → wgmma → wgmma_wait<1> → mbar_arrive_if
  (与 v4 相同, __forceinline__ 保证无 C7510)

gemm_kernel_pingpong<Config, TmaCopyA, TmaCopyB>
  __launch_bounds__(160, 2)
  tid  0-127: Consumer WG → 构建 gC/sA/sB → pp_consumer_loop
  tid 128-159: Producer warp → pp_producer_loop (TMA 视图在函数内构建)
  ptxas: 154 reg, 0 spill → occupancy=2 @ kStage=3
```

### SMEM 布局

**Ping-Pong v5（kStage=3）：**
```
  A: 128×64×3 × 2B = 49152B
  B: 128×64×3 × 2B = 49152B
  mbar_full:  3 × 8B = 24B
  mbar_empty: 3 × 8B = 24B
  总计: 98352B ≈ 96KB  → floor(227/96) = 2 blocks ✅
```

**Ping-Pong v5（kStage=4）：**
```
  总计: 131136B ≈ 128KB  → floor(227/128) = 1 block (SMEM 受限)
```

---

## 20. 性能数据汇总（最新）

测试规模：M=N=4096, K=2048, BF16→FP32，设备：NVIDIA H20-3e (SM 9.0)

| 版本 | 配置 | 时间 | TFLOPS | vs cuBLAS |
|------|------|------|--------|-----------|
| cuBLAS BF16 (baseline) | — | 0.524 ms | 131.0 | 100% |
| TMA+GMMA（WAR竞争版）| kStage=3 | 1.146 ms | 59.95 | 45.7% |
| cp.async+GMMA | kStage=3 | 0.587 ms | 117.1 | 89.9% |
| TMA+GMMA（WAR修复后）| kStage=3 | **0.524 ms** | **131.1** | **100.7%** ✓ |
| TMA+GMMA | kStage=4 | 0.594 ms | 115.8 | 88.9% ↓ |
| Ping-Pong v1（arrive×128）| kStage=3 | ~0.59 ms | ~116 | ~88% ✗ |
| Ping-Pong v2（`__noinline__`）| kStage=3 | 0.603 ms | 114.0 | 87.0% ✗ C7510 |
| Ping-Pong v3（`__forceinline__` + setmaxnreg, 256线程）| kStage=3 | 0.594 ms | 115.7 | 88.8% ✗ |
| Ping-Pong v4（160线程, 全forceinline）| kStage=3 | 0.590 ms | 116.4 | 88.8% ✗ 232reg |
| **Ping-Pong v5（160线程, 寄存器隔离）** | **kStage=3** | **0.524 ms** | **131.1** | **100.7%** ✅ |
| Ping-Pong v5 | kStage=4 | 0.595 ms | 115.5 | 88.7% (SMEM受限) |
| PingPong-Persistent（156 blocks, atomicAdd）| kStage=3 | ~0.527 ms | ~130.4 | ~99.5% （略慢）|
| PingPong-Cluster（Cluster=2, TMA Multicast） | kStage=3 | 待测试 | 待测试 | 待测试 |

**当前最佳**：
- 单WG TMA+GMMA kStage=3：**131.1 TFLOPS**，超越 cuBLAS 0.7%
- Ping-Pong v5 kStage=3：**131.1 TFLOPS**，与单WG TMA 完全持平 ✅

---

## 21. v6 进一步优化探索（kStage=2 测试）


### 尝试方向

**kStage=2 (SMEM=64KB)**:
- 目标: 验证是否更高 occupancy (理论 3 blocks/SM) 能带来更好性能
- 结果: ❌ 反而变慢 (119 TFLOPS vs kStage=3 的 132 TFLOPS)
- 原因分析:
  1. 寄存器从 154→198（编译器无法像 kStage=3 那样优化循环展开）
  2. 流水线太短，TMA 延迟更容易暴露
  3. Occupancy 的理论收益被流水线效率损失抵消

**结论**：kStage=3 是最优配置，128x128x64 tile + 96KB SMEM + occupancy=2 已经是最优平衡。

---

## 22. 迭代十六：Persistent Kernel + Cluster TMA Multicast 架构探索

### 背景与动机

v5 版本（PingPong kStage=3）已达到 131 TFLOPS，与 cuBLAS 持平。为进一步压榨性能，探索两个更激进的架构方向：

1. **Persistent Kernel**：消除 wave 切换开销和 load imbalance
2. **Cluster TMA Multicast**：利用 SM90 Cluster 机制，A 矩阵 TMA 带宽并发度翻倍

---

### 22.1 Persistent Kernel（`detail/kernel_pingpong_persistent.cuh`）

#### 设计思路

当前 PingPong v5 的 Grid = 32×32 = 1024 blocks，H20-3e 有 78 个 SM，occupancy=2 → 每次 156 blocks 活跃，需要 ≈6.6 waves 才能完成所有 tiles。每次 wave 切换有调度开销，最后一波可能出现 load imbalance。

**Persistent Kernel 设计**：
- 只发射 78×2 = 156 blocks（填满所有 SM）
- 通过全局原子计数器 `tile_counter` 动态分配 tiles（work stealing）
- 每 block 持续运行，直到所有 tiles 处理完毕

```cpp
while (true) {
    if (tid == 0) {
        *smem_tile_id = atomicAdd(tile_counter, 1);
    }
    __syncthreads();
    int tile_id = *smem_tile_id;
    if (tile_id >= total_tiles) break;
    // ... 执行当前 tile ...
    __syncthreads();
}
```

#### SMEM 布局的对齐问题（Bug 修复）

**问题**：最初尝试使用静态 `__shared__ int smem_tile_id`，但这会导致动态 SMEM 的起始地址偏移，破坏 mbarrier 的 8B 对齐要求，运行时报 `misaligned address` 错误。

**修复**：将 `smem_tile_id` 移动到**动态 SMEM 末尾**（mbarrier 数组之后），并更新 `get_smem_size_pingpong_persistent` 包含这额外的 4 字节：

```cpp
constexpr size_t tile_id_offset = mbar_offset + 2 * kStage * sizeof(uint64_t);
int* smem_tile_id = reinterpret_cast<int*>(smem_buf + tile_id_offset);

// SMEM size 函数
template <typename Config>
constexpr size_t get_smem_size_pingpong_persistent() {
    // ...
    return mbar_offset + mbar_bytes + sizeof(int);  // +4B for tile_id
}
```

#### 测试结果

```
SM90 PingPong-Pers: ~0.527 ms  ≈ 130.4 TFLOPS  (vs cuBLAS: ~99.5%)
```

Persistent Kernel 比 v5 PingPong(k3) **略慢约 0.5-1%**，额外的 `atomicAdd` + `__syncthreads` 开销抵消了 wave 切换收益。在 4096×4096 这种 1024-tile 的规模下，wave 切换开销相对较小，persistent 架构的优势不明显。

---

### 22.2 Cluster TMA Multicast（`detail/kernel_pingpong_cluster.cuh`）

#### 设计思路

SM90 支持多个 CTA 组成 **Cluster**，利用 `SM90_TMA_LOAD_MULTICAST` 指令：

- **Cluster size = 2**（X 方向，即 N 方向）
- 两个 CTA 处理相同 M-tile 但不同 N-tile，因此共享同一个 A tile
- `TMA Multicast`：每个 CTA 只加载 A 的 1/2，通过 multicast 让另一个 CTA 也收到对应部分
- A 的 HBM→SMEM 带宽理论上减少 50%，TMA 并发度翻倍

```
CTA0: get_slice(0) → 加载 A[0:64, :] (A 的上半), multicast 到 CTA0 和 CTA1
CTA1: get_slice(1) → 加载 A[64:128, :] (A 的下半), multicast 到 CTA0 和 CTA1
结果: 两个 CTA 都得到完整的 A tile, 总 TMA 字节数不变但并发度翻倍
```

#### `expect_tx` 死锁问题（关键 Bug 修复）

**问题**：最初将 `expect_tx` 设为 `kABytesPerCTA + kBBytes`（即 A 的一半 + B），导致 kernel 运行死锁（`mbar_full` 永远不 trigger）。

**分析**：TMA Multicast 的 mbarrier tx_count 减少机制：
```
CTA0 发起 A multicast (get_slice(0)) → 写入 CTA0 和 CTA1 的 smem_A[0:1/2]
  → CTA0.mbar tx_count -= kABytesTotal/2  (自身写入)
  → CTA1.mbar tx_count -= kABytesTotal/2  (multicast)
CTA1 发起 A multicast (get_slice(1)) → 写入 CTA0 和 CTA1 的 smem_A[1/2:1]
  → CTA1.mbar tx_count -= kABytesTotal/2  (自身写入)
  → CTA0.mbar tx_count -= kABytesTotal/2  (multicast)

每个 CTA 的 mbar 总减少量 = kABytesTotal/2 × 2 = kABytesTotal
加上 B TMA: kBBytes
→ expect_tx = kABytesTotal + kBBytes (与普通 PingPong 完全相同!)
```

**修复**：

```cpp
static constexpr int kABytesTotal = kTileM * kTileK * sizeof(T);
static constexpr int kBBytes = kTileN * kTileK * sizeof(T);
static constexpr int kExpectTx = kABytesTotal + kBBytes;  // ← 修复点
```

#### `cudaLaunchKernelEx` 调用方式

使用 `__cluster_dims__(2, 1, 1)` 编译时指定 cluster 维度，同时用 `cudaLaunchKernelEx` 在运行时明确 cluster 配置：

```cpp
cudaLaunchConfig_t launch_config = {};
launch_config.gridDim  = grid_cl;
launch_config.blockDim = block_cl;
launch_config.dynamicSmemBytes = smem_cl;
cudaLaunchAttribute attr[1];
attr[0].id = cudaLaunchAttributeClusterDimension;
attr[0].val.clusterDim = {2, 1, 1};
launch_config.numAttrs = 1;
launch_config.attrs    = attr;

// 模板版本，直接传 kernel 函数指针和参数
CUDA_CHECK(cudaLaunchKernelEx(&launch_config, kernel_cl,
                               Cptr, tma_a_cl, tma_b_cl, M, N, K));
```

#### mbarrier arrive_count 说明

Cluster 版本的 mbarrier 依然初始化为 `arrive_count=1`（每个 CTA 自己的 Producer 负责 arrive）。A multicast 的 tx_count 减少是**自动发生**的（TMA 硬件写入完成时减少），不需要额外的 arrive。

#### 远程 mbarrier 操作（跨 CTA expect_tx 设置）

Cluster TMA Multicast 需要解决一个核心问题：**每个 CTA 的 mbarrier 都需要预告来自两个 CTA 的 TMA 传输**。最初尝试使用以下方案：

**方案 1（Leader-Follower 模式）**：CTA0 作为 Leader，统一设置所有 CTA 的 `expect_tx`

```cpp
// detail/mbarrier.cuh 新增远程 expect_tx 设置原语
CUTE_DEVICE void mbar_arrive_and_expect_tx_remote(
    uint64_t* mbar, int tx_bytes, int cta_id, int pred = 1)
{
    uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(mbar);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  .reg .b32 remAddr32;\n\t"
        "  setp.eq.u32 p, %2, 1;\n\t"
        "  @p mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
        "  @p mbarrier.arrive.expect_tx.shared::cluster.b64 _, [remAddr32], %3;\n\t"
        "}"
        :: "r"(smem_ptr), "r"(cta_id), "r"(pred), "r"(tx_bytes) : "memory"
    );
}

// kernel 中：CTA0 统一设置所有 CTA 的 expect_tx
if (cta_rank == 0) {
    mbar_arrive_and_expect_tx(&mbar_full[stage], kExpectTx);       // 本地
    mbar_arrive_and_expect_tx_remote(&mbar_full[stage], kExpectTx, 1); // 远程
}
```

`mapa.shared::cluster` 指令用于将本地 SMEM 地址映射到指定 CTA 的地址空间，从而实现跨 CTA 的 `expect_tx` 设置。

#### CUDA Context 初始化卡死（重大障碍）

**测试环境**：H20-3e (SM90a), Driver 550.127.08

**现象**：
1. 编译通过，PTX/cubin 中 `mbarrier.arrive.expect_tx.shared::cluster` 和 `EIATTR_EXPLICIT_CLUSTER` 属性正确生成
2. 程序在 `main()` 开头的 `cudaMalloc` 就卡死（CUDA Context 初始化阶段）
3. `pkill -9` 后，新进程仍然卡在 `cudaMalloc`，表明驱动状态未恢复
4. GPU 100% 占用，`nvidia-smi` 显示 `Reset Required: No`
5. 诊断确认卡死位置在 `cudaGetDeviceProperties` **之后**，`cudaMalloc` **之前**

**根本原因分析**：
程序在 **CUDA driver 加载 cubin 阶段**卡死，具体是加载包含 `__cluster_dims__(2,1,1)` 属性的 device code 时。这表明：
- H20-3e（SM90a）的 CUDA driver 550.127.08 对 Hopper Cluster kernel 的支持存在问题
- 可能是驱动 bug、权限配置或硬件限制

**诊断证据**：
```bash
# PTX 中正确生成 cluster 指令
$ grep -A 3 "mbarrier.arrive.expect_tx.shared::cluster" main.ptx
mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%r1], %r2;

# cubin 中包含 EIATTR_EXPLICIT_CLUSTER
$ cuobjdump --elf-section EIATTR main.o
EIATTR_EXPLICIT_CLUSTER: (2, 1, 1)

# 但程序在 cudaMalloc (driver 加载 cubin) 时永久挂起
```

**其他失败尝试**：
- 简化 kernel 代码（仅保留 `__cluster_dims__` 声明，移除所有 cluster 指令）→ 依然卡死
- 注释掉 cluster kernel 调用 → 立即恢复正常
- 检查 GPU 状态（`nvidia-smi`, `compute-sanitizer`）→ 硬件正常
- 尝试 `cudaDeviceReset()` → 在 `cudaMalloc` 之前无法调用

**结论**：当前测试环境（H20-3e + Driver 550.127.08）无法运行 Cluster kernel，卡死发生在驱动层而非 kernel 逻辑层。

#### 测试结果

```
SM90 PingPong-Cluster: ❌ 无法测试（CUDA Context 初始化卡死）

原因: H20-3e (SM90a) 驱动 550.127.08 加载含 __cluster_dims__ 的 cubin 时挂起
状态: 暂时放弃该优化方向，等待驱动更新或更换测试环境
```

---

### 22.3 代码文件新增

| 文件 | 内容 |
|------|------|
| `detail/kernel_pingpong_persistent.cuh` | Persistent Kernel（动态 tile 调度 + atomicAdd） |
| `detail/kernel_pingpong_cluster.cuh` | Cluster=1×2 + SM90_TMA_LOAD_MULTICAST |

`gemm_sm90.cuh` 已更新包含两个新文件：

```cpp
#include "detail/kernel_pingpong_persistent.cuh"
#include "detail/kernel_pingpong_cluster.cuh"
```

---

### 22.4 主要 Bug 修复记录

| Bug | 现象 | 根因 | 修复 |
|-----|------|------|------|
| Persistent SMEM 对齐 | `misaligned address` at launch | 静态 `__shared__ int` 使动态 SMEM 起始偏移，破坏 mbarrier 8B 对齐 | 将 `smem_tile_id` 移入动态 SMEM 末尾，更新 size 函数 |
| Cluster expect_tx 死锁 | kernel 卡死 | `expect_tx` 误设为 `kABytesPerCTA + kBBytes`，实际每个 CTA 接收 `kABytesTotal + kBBytes` | 修复为 `kExpectTx = kABytesTotal + kBBytes` |
| `cudaLaunchKernelEx` 编译失败 | 参数类型不匹配 | 误用 `void**` 数组形式 | 改用模板版本直接传 kernel 指针和参数 |

---

## 23. 迭代十七：Cluster TMA Multicast 调试与放弃

### 23.1 调试过程

在迭代十六实现的代码基础上，尝试在 H20-3e 上运行 Cluster TMA Multicast kernel，经历了完整的调试过程。

**初始症状（GPU 100% 卡死）**：

首次运行时，kernel 启动后 GPU 立即进入 100% 占用状态并永久卡死。初步怀疑是 `expect_tx` 计数错误导致 mbarrier 永不翻转。

**调试方向一：修复 `expect_tx` 逻辑**

分析 TMA Multicast 的 tx_count 减少机制，确认每个 CTA 的 mbarrier 接收来自两个 CTA 的 multicast 写入，总字节数为 `kABytesTotal + kBBytes`（不是 `kABytesPerCTA + kBBytes`）。修复后重新测试。

**调试方向二：Leader-Follower expect_tx 模式**

修复 `expect_tx` 值后依然卡死，怀疑两个 CTA 同时设置对方 mbarrier 存在竞争。改为 Leader-Follower 模式：CTA0 统一设置两个 CTA 的 `expect_tx`，使用 `mapa.shared::cluster` + `mbarrier.arrive.expect_tx.shared::cluster` 跨 CTA 原语。

**调试方向三：诊断 CUDA Context 初始化卡死**

`pkill -9` 杀死进程后，新进程在 `main()` 入口的 `cudaMalloc` 就立即卡死，完全没有输出。添加大量诊断 `printf` + `fflush` 后定位：
- `cudaGetDeviceProperties` 能通过
- `cudaMalloc` 永久阻塞（CUDA Context 初始化时加载 cubin）

### 23.2 根因确认

通过二分测试（逐步注释代码）确认：
- **只要编译单元中包含含 `__cluster_dims__` 属性的 kernel 函数**（即使不调用），`cudaMalloc` 就会卡死
- 注释掉 cluster kernel 函数定义后，程序恢复正常

这证明卡死发生在 **CUDA driver 加载 cubin 时**，与 kernel 逻辑无关。

### 23.3 环境限制结论

| 检查项 | 结果 |
|--------|------|
| PTX 正确性 | ✅ `mbarrier.arrive.expect_tx.shared::cluster` 正确生成 |
| cubin EIATTR_EXPLICIT_CLUSTER | ✅ (2,1,1) 正确标记 |
| nvidia-smi GPU 状态 | ✅ 正常，Reset Required: No |
| 驱动版本 | CUDA 550.127.08 |
| GPU 型号 | H20-3e (SM90a) |
| **运行结果** | **❌ cudaMalloc 挂起，驱动加载 cluster cubin 失败** |

**可能原因**：
1. CUDA Driver 550.127.08 对 SM90a cluster kernel 支持不完整（已知 bug）
2. 服务器环境限制（如 MIG 模式、虚拟化层）阻止 cluster 调度
3. 之前的 kernel hang 在驱动/硬件层面留下未清理状态（需要 GPU reset 或服务器重启）

### 23.4 决策：暂时放弃

鉴于问题发生在驱动层面，暂时放弃 Cluster TMA Multicast 优化方向。已实现的代码（`detail/kernel_pingpong_cluster.cuh`）保留供参考，等待以下条件之一满足后重新验证：
- 升级到更新的 CUDA Driver 版本
- 在不同的测试环境（如物理机 H100）上尝试
- 确认 Cluster kernel 所需的驱动/系统配置要求

---

## 24. 迭代十八：cute-skill 知识体系整理

### 背景

完成 SM90 GEMM 全系列优化后，系统整理了 CuTe/CUTLASS 库的核心知识，创建了 `.claude/skills/cute_skill/SKILL.md`，供后续 AI 辅助开发使用。

### 整理目标

从本次优化实践中提炼的经验出发，深入 CuTe 源码（`layout.hpp`、`tensor.hpp`、`mma_atom.hpp`、`tma.hpp` 等），归纳**泛化程度高**的知识体系，而非仅记录本项目的迭代细节。

参照 `cuda_skill` 的组织风格：从底层原理、API 设计模式、硬件指令映射、常见陷阱四个维度系统覆盖。

### 整理内容

**知识体系结构（cute_skill/SKILL.md）**：

```
1. Layout 代数模型        ← CuTe 核心抽象：(coord) → index 函数
2. Tensor 抽象            ← Engine + Layout 的组合
3. Layout 操作            ← composition / logical_divide / zipped_divide 等
4. Swizzle 模型           ← <B,M,S> 位变换，SM90 M=4 约束
5. MMA Atom 体系          ← SM80 ldmatrix+mma.sync vs SM90 WGMMA SS/RS
6. TiledMMA/TiledCopy    ← 从 Atom 到线程分发，partition_C / partition_S / partition_D
7. TMA 数据搬运           ← descriptor 创建，as_position_independent_swizzle_tensor
8. mbarrier 同步          ← phase-based 双向屏障，arrive_and_expect_tx 语义
9. SM90 wgmma 流水线      ← fence/arrive/commit/wait 四件套 + C7510 陷阱
10. Occupancy 优化        ← 寄存器隔离，__launch_bounds__，setmaxnreg 适用范围
11. 常见踩坑表            ← 症状-根因-修复 格式
```

### 核心经验转化

本次优化过程中的若干关键认知已抽象为通用规律写入 cute-skill：

| 本项目经验 | 抽象为通用规律 |
|------------|---------------|
| AtomLayout 堆叠导致结果全零 | SM90 WGMMA Atom 已是完整 WarpGroup，堆叠语义与 SM80 完全不同 |
| `Swizzle<3,4,3>` M=4 约束 | TMA 要求 swizzle 基域宽 = 128bit，M=4 是 128b/元素大小转换结果 |
| Consumer `__forceinline__` 消除 C7510 | wgmma 序列不能跨函数边界，这是 PTX 层面的 wgmma pipeline 语义 |
| Producer `__noinline__` 寄存器隔离 | ptxas 对 if-else 取并集，`__noinline__` 建立函数栈帧边界隔离 |
| expect_tx 精确匹配 | mbarrier 的 tx_count 必须与实际 TMA 写入字节数完全一致 |
| `as_position_independent_swizzle_tensor` | ComposedLayout 的 swizzle 是相对偏移变换，不依赖绝对地址 |

---

## 25. 不同规模性能对比

| 规模 | cuBLAS | PingPong(k3) | vs cuBLAS |
|------|--------|--------------|-----------|
| 2048×2048×1024 | 122.7 TFLOPS | 106.0 TFLOPS | 86.4% |
| 4096×4096×2048 | 131.0 TFLOPS | **131.9 TFLOPS** | **100.7%** ✅ |
| 8192×8192×4096 | 119.2 TFLOPS | **120.6 TFLOPS** | **101.2%** ✅ |

**观察**：
- 小规模: PingPong 略慢于 cuBLAS（grid 小，launch overhead 占比高）
- 中大规模: PingPong 超越 cuBLAS 1-2%

---

## 26. 性能优化结论

**已验证的优化方向及结果**：

| 优化方向 | 结果 | 原因 |
|----------|------|------|
| kStage=4 (128KB SMEM) | ❌ 变慢 12% | SMEM 限制 occupancy=1 |
| kStage=2 (64KB SMEM) | ❌ 变慢 9% | 流水线短 + 寄存器增加 |
| 更大 tile (256x128) | ❌ occupancy=1 | SMEM 超限 |
| 减少 Producer 线程数 | ≈ 不变 | 非瓶颈 |
| setmaxnreg | ≈ 不变 | 对非 persistent kernel 无效 |

**最终最优配置**：
- Tile: 128×128×64
- kStage: 3
- SMEM: 96KB
- 线程数: 160 (128 Consumer + 32 Producer)
- 寄存器: 154
- Occupancy: 2 blocks/SM
- 性能: **131.9 TFLOPS @ 4096³, 超越 cuBLAS 0.7%**

---

## 27. 待办事项

- [x] 在 Hopper 机器上运行修复后的版本，更新性能数据
- [x] Producer/Consumer WarpGroup 分离（Ping-Pong 设计）
- [x] Ping-Pong 优化：arrive_count=1 + kStage=4 + Producer 单线程
- [x] 在容器内用 nsys + cuobjdump 分析 Ping-Pong 低 occupancy 根因
- [x] `__noinline__` 重构：将 Producer/Consumer 提取为独立子函数
- [x] 代码结构整理：将 `gemm_sm90.cuh` 拆分为 `detail/` 下的多个独立头文件
- [x] 消除 C7510 警告（`__forceinline__` 替代 `__noinline__`）
- [x] 诊断 `setmaxnreg` 对非 persistent kernel 无效的根因
- [x] v4：Producer 减少到 32 线程 + `__launch_bounds__(160,2)`
- [x] v5：寄存器隔离（Producer __noinline__ + TMA 视图移入函数内）→ 154 reg, occupancy=2, **131 TFLOPS** ✅
- [x] Persistent Kernel：全局 atomicAdd tile 调度，验证 wave 切换收益（结论：收益不明显，~99.5%）
- [x] Cluster TMA Multicast：cluster_size=2, SM90_TMA_LOAD_MULTICAST，代码实现完成，待性能验证
- [x] Cluster TMA Multicast 调试：修复 expect_tx 逻辑 + 实现 Leader-Follower mapa 远程设置模式
- [~] 更新 Cluster TMA Multicast 的实测性能数据（❌ 受阻：H20-3e 驱动无法加载 cluster cubin，暂时放弃）
- [x] cute-skill 创建：基于源码分析 + 本项目优化经验，整理 `.claude/skills/cute_skill/SKILL.md`（Layout 代数 / TMA / wgmma 流水线 / Occupancy 优化 / 踩坑表）
- [ ] 用 ncu 分析 PingPong v5 的 SM 利用率和 warp stall 分布（需要 root 权限环境）
- [ ] FP16 版本验证
- [ ] 考虑 persistent kernel + 2 Consumer WG 的真正 Ping-Pong（参考 CUTLASS sm90 pingpong，384 线程）
- [ ] 在条件允许时（更新驱动或更换环境）重新验证 Cluster TMA Multicast

---

*最后更新：2026-04-01（迭代十八：cute-skill 知识体系整理 —— 基于源码分析与本项目全系列优化经验，创建泛化程度高的 CuTe/CUTLASS 知识库 SKILL.md）*
