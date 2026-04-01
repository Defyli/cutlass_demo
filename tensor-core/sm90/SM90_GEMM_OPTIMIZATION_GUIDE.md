# SM90 (Hopper) GEMM 手写优化经验总结

> 本文是在 H20-3e (SM90a) 上从零实现 BF16 GEMM 算子，达到并超越 cuBLAS 性能的完整经验提炼。
> 面向读者：需要在 Hopper 架构上手写高性能 GEMM 的 CUDA 工程师。

---

## 目录

1. [Hopper 架构关键差异（vs Ampere）](#1-hopper-架构关键差异vs-ampere)
2. [mbarrier 使用要点](#2-mbarrier-使用要点)
3. [TMA 使用要点](#3-tma-使用要点)
4. [wgmma (GMMA) 使用要点](#4-wgmma-gmma-使用要点)
5. [软件流水线（kStage）设计](#5-软件流水线kstage设计)
6. [Producer-Consumer 分离（Ping-Pong）](#6-producer-consumer-分离ping-pong)
7. [寄存器控制与 Occupancy 优化](#7-寄存器控制与-occupancy-优化)
8. [SMEM 布局与对齐](#8-smem-布局与对齐)
9. [Tile 尺寸与流水线深度选择](#9-tile-尺寸与流水线深度选择)
10. [Persistent Kernel 适用场景](#10-persistent-kernel-适用场景)
11. [Cluster TMA Multicast（环境限制）](#11-cluster-tma-multicast环境限制)
12. [性能调试工具与方法](#12-性能调试工具与方法)
13. [经验结论速查表](#13-经验结论速查表)
14. [不同规模性能参考](#14-不同规模性能参考)

---

## 1. Hopper 架构关键差异（vs Ampere）

| 特性 | Ampere (SM80) | Hopper (SM90) |
|------|--------------|--------------|
| MMA 指令 | `mma.sync`（32线程/warp） | `wgmma.mma_async`（128线程/WarpGroup） |
| MMA 形状 | 16×8×16 | 64×N×16（N=8,16,...,256） |
| G→S 数据搬运 | `cp.async`（多线程分工） | TMA（**1线程**发起整块） |
| 同步原语 | `cp_async_fence/wait` | `mbarrier`（phase-based双向屏障） |
| SMEM→Reg | `ldmatrix` 显式搬运 | SS模式：`wgmma` 直接读SMEM，**无需ldmatrix** |
| SMEM Swizzle | `Swizzle<3,3,3>` | `Swizzle<3,4,3>`（**M参数必须=4**） |
| 累加器格式 | FP16 或 FP32 | FP32（BF16/FP16输入固定FP32累加） |
| CTA 协作 | 无 | Cluster（多个CTA共享SMEM数据） |
| 动态寄存器 | 无 | `setmaxnreg`（persistent kernel专用） |
| 最大 SMEM/SM | 164KB | **227KB**（需显式申请） |

**TiledMMA 配置关键点**：

```cpp
// ✅ 推荐：用 ss_op_selector 自动根据元素类型 + TileShape 选择最优 MMA Op
// 头文件：#include "cute/arch/mma_sm90.hpp"
using MMA_Op = decltype(GMMA::ss_op_selector<
    bfloat16_t,                              // ElementA
    bfloat16_t,                              // ElementB
    float,                                   // ElementC (accumulator)
    Shape<_128, _128, _64>,                  // TileShape_MNK
    GMMA::Major::K,                          // MajorA
    GMMA::Major::K                           // MajorB
>());
using TiledMMA = decltype(make_tiled_mma(MMA_Atom<MMA_Traits<MMA_Op>>{}));
// 换 Tile 或元素类型时不需要手动查表，编译期自动匹配正确的 SM90_64xNxK_... 类型

// ✅ 也可手动硬编码（Tile 固定时）
using MMA_Op = SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>;

// ❌ 错误：AtomLayout 堆叠会要求 256 线程，但 kernel 只启动 128
using TiledMMA = decltype(make_tiled_mma(MMA_Atom<...>{},
    Layout<Shape<_2,_1,_1>>{}));  // ← 危险，会导致结果为零或越界
```

---

## 2. mbarrier 使用要点

### 状态模型

```
state = phase(1bit) | arrive_count(20bit) | tx_count(21bit)
```

- `arrive_count` 降至 0 且 `tx_count` 降至 0 时，`phase` 自动翻转（0→1→0…）
- 两个方向的屏障：
  - `mbar_full[stage]`：通知 Consumer "数据已就绪"（TMA 写完后触发）
  - `mbar_empty[stage]`：通知 Producer "Consumer 已消费，可以覆写"

### 初始化

```cpp
// Ping-Pong 设计下：arrive_count=1（只需 Producer 线程 arrive 一次）
mbar_init(&mbar_full[s],  1);
mbar_init(&mbar_empty[s], 1);

// ⚠️ 关键：mbar_empty 需要预先 arrive kStage 次（表示初始阶段 SMEM 均为空）
for (int s = 0; s < kStage; ++s) {
    mbar_arrive(&mbar_empty[s]);  // 使 empty 屏障立即可等待
}
```

### Producer 端正确写法

```cpp
// ✅ 顺序：先 wait empty → 再 arrive_and_expect_tx → 再发 TMA copy
mbar_wait(&mbar_empty[stage], phase_e);
mbar_arrive_and_expect_tx(&mbar_full[stage], expect_bytes);
tma_copy(...);  // 硬件写完后自动将 mbar_full.tx_count 减至 0
```

### `expect_tx` 字节数计算

`expect_tx` 必须与 TMA **实际写入该 mbarrier 对应 SMEM 区域的字节数**精确匹配：

```cpp
// 标准 PingPong（A + B 各一 tile）
constexpr int kExpectTx = kTileM * kTileK * sizeof(T)   // A tile
                        + kTileN * kTileK * sizeof(T);  // B tile

// ⚠️ 常见错误：误用 kABytesPerCTA（仅 A 的一半）
// 这会导致 mbar_full 的 tx_count 永远不归零 → kernel 死锁
```

### WAR（Write-After-Read）数据竞争

**必须在每次 stage 迭代末尾 arrive `mbar_empty`**，表示 Consumer 已读完当前 stage，允许 Producer 在下一个循环覆写：

```cpp
// Consumer 循环末尾（每 k_tile 迭代）
mbar_arrive(&mbar_empty[stage]);  // ← 缺少这行 → 结果错误或死锁
```

### mbarrier 对齐要求

`mbarrier` 变量必须 **8 字节对齐**：
- 将 `mbarrier` 数组放在动态 SMEM 的最开头（最安全）
- 若有 `__shared__` 静态变量，会使动态 SMEM 起始地址偏移，破坏对齐
  → **解决方案**：将所有辅助变量（如 `tile_id`）移入动态 SMEM 末尾

---

## 3. TMA 使用要点

### TMA descriptor 创建

```cpp
// Host 端：make_tma_copy 创建 TMA descriptor（通过模板参数传入 kernel）
auto tma_a = make_tma_copy(
    SM90_TMA_LOAD{},
    ptr_A,                           // 原始 device 指针
    typename Config::GmemLayoutA{},  // GMEM 布局（含 Swizzle）
    typename Config::SmemLayoutA{},  // SMEM 布局
    cute::Int<1>{}                   // cluster_size（非 multicast 填 1）
);
```

### TMA Multicast

```cpp
// multicast 版本：cluster_size=2，两个 CTA 共享同一 A tile
auto tma_a_multicast = make_tma_copy(
    SM90_TMA_LOAD_MULTICAST{},
    ptr_A,
    gmem_layout_A,
    smem_layout_A,
    cute::Int<2>{}  // cluster_size
);
```

在 kernel 中需要指定 `multicast_mask`（哪些 CTA 接收数据）：

```cpp
uint16_t multicast_mask = (1 << cluster_size) - 1;  // = 0b11 for cluster_size=2
tma_copy_multicast(..., multicast_mask);
```

### TMA 发起只需 1 个线程

```cpp
if (tid == 0) {   // 或 prod_tid == 0
    tma_copy(...);
}
// 其他线程不参与 TMA 发起，可以执行其他工作或直接 spin-wait
```

---

## 4. wgmma (GMMA) 使用要点

### 正确执行序列

```cpp
wgmma_fence();    // 确保 SMEM 写入对 wgmma 可见（防止 RAW 竞争）
wgmma_arrive();   // 开启 wgmma 异步执行组
gemm(tiled_mma, tCrA_view, tCrB_view, tCrC);  // 提交 wgmma 指令
wgmma_commit();   // 提交本轮 wgmma group
wgmma_wait<1>();  // 等待到只剩 1 个 in-flight group（允许流水线）
// ⚠️ 最后一个 tile 要 wgmma_wait<0>()，确保结果完全写回
```

### C7510 警告（wgmma pipeline 序列化）

```
warning #7510-D: wgmma.mma_async pipeline: wgmma pipeline may be suboptimal
  due to crossing function boundary
```

- **根因**：包含 `wgmma` 的函数被 `__noinline__` 标注，编译器在函数边界插入隐式 `wgmma.wait_group 0`，序列化整个 GMMA 流水线
- **修复**：含 `wgmma` 的 Consumer 函数必须用 `__forceinline__`

---

## 5. 软件流水线（kStage）设计

### 流水线预热

在主循环前，必须预填 `kStage-1` 个 stage：

```cpp
// 预热：填满流水线
for (int s = 0; s < kStage - 1; ++s) {
    mbar_wait(&mbar_empty[s], 0);
    mbar_arrive_and_expect_tx(&mbar_full[s], kExpectTx);
    tma_copy(stage = s);
    advance_k();
}
// 主循环
for (int tile = kStage-1; tile < total_k_tiles; ++tile) {
    // Producer：发射下一 tile
    // Consumer：消费最旧 tile
}
// Drain：清空已发射但未消费的 stage
for (int s = 0; s < kStage - 1; ++s) {
    // Consumer 消费剩余 stage
}
```

### kStage 取值权衡

| kStage | SMEM (128×128×64 tile) | occupancy | 结论 |
|--------|------------------------|-----------|------|
| 2 | 64KB | 3 blocks 理论 | ❌ 实际慢 9%，流水线过短 + 寄存器增加 |
| **3** | **96KB** | **2 blocks** ✅ | **最优，TMA 延迟充分隐藏** |
| 4 | 128KB | 1 block（SMEM 受限） | ❌ 慢 12%，occupancy 降为 1 |

**结论**：对于 128×128×64 tile，`kStage=3` 是最优配置。

---

## 6. Producer-Consumer 分离（Ping-Pong）

### 设计原理

**问题**：单 WarpGroup 设计中，Consumer 执行 `wgmma_wait<1>` 时等待上一组 MMA 完成，期间 SM 闲置。

**Ping-Pong 思路**：将 128 线程中的一部分划出作为专职 Producer（专门做 TMA），另一部分作为 Consumer（专门做 MMA），通过 mbarrier 同步。

### 线程划分（最终方案 v5）

```
总线程数 = 160（使用 __launch_bounds__(160, 2)）
  tid   0-127: Consumer WarpGroup（128线程，参与 wgmma）
  tid 128-159: Producer warp（32线程，只有 tid==128 发起 TMA）
```

**为什么 Producer 只需 32 线程（1 warp）**：
- TMA 发起只需 1 个线程（`prod_tid==0`），其余空转
- mbarrier arrive/wait 也只需 1 个线程
- 减少 Producer 线程数 = 减少总寄存器用量 = 提升 occupancy

---

## 7. 寄存器控制与 Occupancy 优化

### 关键公式

```
occupancy = min(
    floor(65536 / (reg_per_thread × threads_per_block)),  // 寄存器限制
    floor(smem_per_sm / smem_per_block)                   // SMEM 限制
)
```

H20-3e: 寄存器文件 = 65536，最大 SMEM = 227KB

### `__launch_bounds__` 的双重作用

```cpp
__launch_bounds__(160, 2)
// 参数1: maxThreadsPerBlock = 160（告知 ptxas 实际线程数）
// 参数2: minBlocksPerMultiprocessor = 2（要求 ptxas 将寄存器限制在 floor(65536/(2×160)) = 204）
// → ptxas 据此优化至 154 reg（自然满足 occupancy=2）
```

### `setmaxnreg` 只对 Persistent Kernel 有效

```
普通 kernel 启动流程：
  SM 驻留 block ← 按 ptxas 静态值（160reg × 160thread = 25600）预分配
  → setmaxnreg 在 block 入驻后执行，此时 SM 已无法增加驻留数
  → setmaxnreg 对普通 kernel 无收益

Persistent kernel 中：
  setmaxnreg.dec（WG 完成工作后释放寄存器）
  → SM 动态调度新 block 进入（此时物理寄存器已释放）
  → setmaxnreg 有效
```

### 寄存器隔离：Producer `__noinline__` + Consumer `__forceinline__`

**问题根源**：Producer 和 Consumer 均 `__forceinline__` 时，ptxas 为整个 kernel 同时分配两路寄存器：

```
Consumer: ~154 reg（wgmma 累加器 + tCrC 128 FP32）
Producer: ~78  reg（TMA 张量视图 tAgA/tAsA/tBgB/tBsB）
合并后:    232 reg（无法达到 occupancy=2）
```

**解决方案（v5）**：

```
Consumer 路径: __forceinline__（wgmma 不能有函数边界，否则 C7510）
Producer 路径: __noinline__ + TMA 视图在函数内部构建
```

关键点：**TMA 张量视图（`tAgA`, `tAsA` 等）必须在 Producer 函数内部创建**，不能作为参数从 kernel 传入。否则这些寄存器仍会出现在 kernel 本体的 Consumer 代码路径中。

```cpp
// ✅ 正确：TMA 视图在 __noinline__ 函数内构建，寄存器不泄漏到 kernel 本体
__device__ __noinline__ void pp_producer_loop(
    char* smem_A_ptr, char* smem_B_ptr, int bx, int by, ...)
{
    if (prod_tid == 0) {
        // TMA 视图在此构建（寄存器限制在本函数栈帧）
        auto mA = tma_a.get_tma_tensor(...);
        auto tAgA = cta_tma.partition_S(gA);
        // ... TMA copy loop ...
    }
}

// ❌ 错误：TMA 视图在 kernel 本体构建后传入，寄存器仍污染 kernel 本体
auto tAgA = cta_tma.partition_S(gA);  // ← 这行在 kernel 本体 = Consumer 也看到这些 reg
pp_producer_loop(tAgA, ...);
```

### 寄存器优化迭代历史（速查）

| 版本 | 策略 | 寄存器 | Occupancy | TFLOPS |
|------|------|--------|-----------|--------|
| v1 | 256线程，arrive×128 | — | 1 | ~116 |
| v2 | `__noinline__` 分离 | ~160 | 1 | 114（C7510 序列化） |
| v3 | `__forceinline__` + setmaxnreg，256线程 | 160 | 1 | 116（setmaxnreg 无效） |
| v4 | `__forceinline__`，160线程 | **232** | 1 | 116（TMA视图污染） |
| **v5** | Consumer `__forceinline__` + Producer `__noinline__`（TMA视图内构建），160线程 | **154** | **2** ✅ | **131** ✅ |

---

## 8. SMEM 布局与对齐

### Swizzle 参数与自动选择

SM90 要求使用 `Swizzle<3,4,3>`（M参数=4），而非 Ampere 的 `Swizzle<3,3,3>`。

**推荐：用 `ss_smem_selector` 自动选择最优 Swizzle**，避免手动查表：

```cpp
// 头文件：#include "cutlass/gemm/collective/builders/sm90_common.inl"
// namespace: cutlass::gemm::collective::detail

// A 矩阵 (K-major)：根据 kTileK 自动选择最优的 Swizzle 等级
using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    GMMA::Major::K,   // K-major layout
    bfloat16_t,       // 元素类型
    Int<kTileM>,      // BLK_MN
    Int<kTileK>       // BLK_K
>());

// B 矩阵同理
using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    GMMA::Major::K, bfloat16_t, Int<kTileN>, Int<kTileK>
>());
```

**选择逻辑（K-major，BF16/FP16）**：

| kTileK 整除条件 | 返回的 Atom | Swizzle | 说明 |
|-----------------|-------------|---------|------|
| `% 64 == 0` | `Layout_K_SW128_Atom` | `Swizzle<3,4,3>` | 最优，128bit swizzle |
| `% 32 == 0` | `Layout_K_SW64_Atom`  | `Swizzle<3,3,3>` | 次优 |
| `% 16 == 0` | `Layout_K_SW32_Atom`  | `Swizzle<2,3,3>` | 再次 |
| else          | `Layout_K_INTER_Atom` | 无 swizzle | bank conflict 较多 |

对于标准的 `kTileK=64`，`ss_smem_selector` 会自动返回 `Layout_K_SW128_Atom<T>`，等价于手写 `Swizzle<3,4,3>`。

**手动硬编码（等价写法，Tile 固定时）**：

```cpp
using SmemLayoutAtom = decltype(composition(
    Swizzle<3, 4, 3>{},       // ← SM90 必须 M=4（对应 SW128）
    Layout<Shape<_8, _64>, Stride<_64, _1>>{}
));
// 或直接使用命名类型
using SmemLayoutAtomA = GMMA::Layout_K_SW128_Atom<bfloat16_t>;
```

### mbarrier 对齐

- `mbarrier` 需要 8 字节对齐
- 在动态 SMEM 中，mbarrier 数组应放在**最开头**
- `__shared__` 静态变量会使动态 SMEM 偏移，破坏对齐

```cpp
// ✅ 正确布局（动态 SMEM 内）
// [smem_A][smem_B][mbar_full...][mbar_empty...][tile_id(可选)]
constexpr size_t smem_A_offset    = 0;
constexpr size_t smem_B_offset    = kSmemABytes;
constexpr size_t mbar_full_offset = smem_B_offset + kSmemBBytes;
constexpr size_t mbar_empty_offset= mbar_full_offset + kStage * 8;
// 若需要 tile_id（persistent kernel）：放在末尾，避免影响 mbarrier 对齐
constexpr size_t tile_id_offset   = mbar_empty_offset + kStage * 8;
```

### 申请超过 48KB 的 SMEM

```cpp
// 编译时：kernel 内声明（或用 __maxnreg__ 等 attribute）
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
// 并在 launch 时传入 smem_size
```

---

## 9. Tile 尺寸与流水线深度选择

### 最优配置（H20-3e, BF16 GEMM）

```
Tile:      128 × 128 × 64  （M × N × K）
kStage:    3
SMEM:      96KB / block
线程数:    160（128 Consumer + 32 Producer）
寄存器:    154 / thread
Occupancy: 2 blocks / SM
性能:      131 TFLOPS（超越 cuBLAS 0.7%）
```

### Tile 尺寸选择原则

1. **M、N 维度**：应使 wgmma 充分填充 SM 算力（通常 128×128）
2. **K 维度**：64 是 SM90 BF16 wgmma 的最小单元，足够隐藏 TMA 延迟
3. **SMEM 限制**：kStage=3 时 SMEM = 2 × TileM×TileK×2 × kStage = 96KB，允许 2 blocks/SM

### 大 Tile 的代价

- 256×128 tile → SMEM 超限（>128KB/block）→ occupancy=1，反而更慢
- kStage=4（128KB）→ occupancy=1（SMEM 受限）→ 慢 12%

---

## 10. Persistent Kernel 适用场景

### 设计模式

```cpp
// 发射数量 = SM数 × occupancy（填满所有 SM）
int num_sms = prop.multiProcessorCount;  // H20: 78
dim3 grid(num_sms * occupancy);          // = 156

// Kernel 内：work stealing
while (true) {
    if (tid == 0) *smem_tile_id = atomicAdd(tile_counter, 1);
    __syncthreads();
    if (*smem_tile_id >= total_tiles) break;
    // 执行 tile ...
    __syncthreads();  // 确保所有线程完成当前 tile 再 steal 下一个
}
```

### 收益与限制

**收益**：
- 消除多 wave 之间的 launch overhead
- 动态 load balancing（避免最后一 wave 的 idle SM）

**实测结论**（H20-3e, 4096×4096）：
- 1024 tiles / 156 blocks ≈ 6.6 waves，wave 切换开销 **< 0.5%**
- Persistent kernel：~130.4 TFLOPS vs 普通 kernel：~131.1 TFLOPS（**略慢**）
- `atomicAdd` + `__syncthreads` 的额外开销抵消了收益

**适用场景**：问题规模小（tiles 数量 ≤ 3-4 waves）时，Persistent 收益才明显。对于 GEMM 通常不值得引入额外复杂度。

### SMEM 对齐陷阱（Persistent Kernel 特有）

Persistent Kernel 需要在动态 SMEM 中存储 `tile_id` 计数器，**必须放在动态 SMEM 末尾**而非静态 SMEM：

```cpp
// ❌ 错误：使用 __shared__ 静态变量
__shared__ int smem_tile_id;  // 导致动态 SMEM 起始地址偏移，破坏 mbarrier 8B 对齐 → misaligned address

// ✅ 正确：将 tile_id 放在动态 SMEM 末尾
constexpr size_t tile_id_offset = mbar_offset + 2 * kStage * sizeof(uint64_t);
int* smem_tile_id = reinterpret_cast<int*>(smem_buf + tile_id_offset);

// SMEM size 必须包含这额外 4 字节
template <typename Config>
constexpr size_t get_smem_size_persistent() {
    return mbar_offset + 2 * Config::kStage * sizeof(uint64_t) + sizeof(int);  // +4B tile_id
}
```

**根因**：`__shared__` 静态变量会占用静态 SMEM，使动态 SMEM 起始地址从对齐的地址偏移，而 mbarrier 必须 8 字节对齐。将所有辅助变量放到动态 SMEM 末尾是最安全的做法。

---

## 11. Cluster TMA Multicast（环境限制）

### 设计原理

SM90 Cluster 允许多个 CTA 协作共享数据：
- Cluster size = 2（X/N 方向）
- CTA0 和 CTA1 处理相同 M-tile 但不同 N-tile，因此共享同一 A tile
- `SM90_TMA_LOAD_MULTICAST`：每个 CTA 加载 A 的一半，通过硬件 multicast 让两个 CTA 都收到完整 A
- 理论上 A 的 HBM→SMEM TMA 并发度翻倍（带宽压力减半）

### `expect_tx` 计算（易错点）

TMA Multicast 中，每个 CTA 的 mbarrier 会收到来自**两个 CTA 的 multicast 写入**：

```
CTA0 发起 A multicast（A 上半）→ 写入 CTA0.smem_A[0:1/2] 和 CTA1.smem_A[0:1/2]
CTA1 发起 A multicast（A 下半）→ 写入 CTA1.smem_A[1/2:1] 和 CTA0.smem_A[1/2:1]

每个 CTA 的 mbar_full.tx_count 总减少量：
  = kABytesTotal / 2 × 2 + kBBytes
  = kABytesTotal + kBBytes

// ✅ expect_tx 与普通 PingPong 完全相同！
constexpr int kExpectTx = kABytesTotal + kBBytes;

// ❌ 常见错误：以为只有一半 A，设为 kABytesPerCTA + kBBytes
// → mbar_full.tx_count 永不归零 → kernel 死锁
```

### 跨 CTA mbarrier 操作（`mapa` 指令）

Leader-Follower 模式：CTA0 统一设置所有 CTA 的 `expect_tx`，避免两 CTA 并发竞争：

```cpp
// detail/mbarrier.cuh
CUTE_DEVICE void mbar_arrive_and_expect_tx_remote(
    uint64_t* mbar, int tx_bytes, int cta_id, int pred = 1)
{
    uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(mbar);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  .reg .b32 remAddr32;\n\t"
        "  setp.eq.u32 p, %2, 1;\n\t"
        "  @p mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"  // 映射远端地址
        "  @p mbarrier.arrive.expect_tx.shared::cluster.b64 _, [remAddr32], %3;\n\t"
        "}"
        :: "r"(smem_ptr), "r"(cta_id), "r"(pred), "r"(tx_bytes) : "memory"
    );
}
```

### Cluster TMA Multicast 调试流程与关键 Bug

#### 典型一：expect_tx 死锁

**症状**：Kernel 启动后 GPU 100% 占用并永久卡死。

**根因**：将 `expect_tx` 误设为 `kABytesPerCTA + kBBytes`（A 的一半 + B）。实际 TMA Multicast 中每个 CTA 的 mbarrier 接收来自**两个 CTA** 的 multicast 写入：

```
CTA0 发起 A multicast → 写入 CTA0.smem_A[0:1/2] 和 CTA1.smem_A[0:1/2]
  → CTA0.mbar.tx_count -= kABytesTotal/2  |  CTA1.mbar.tx_count -= kABytesTotal/2
CTA1 发起 A multicast → 写入 CTA0.smem_A[1/2:1] 和 CTA1.smem_A[1/2:1]
  → CTA0.mbar.tx_count -= kABytesTotal/2  |  CTA1.mbar.tx_count -= kABytesTotal/2

每个 CTA 的 mbar.tx_count 总减少量 = kABytesTotal + kBBytes
// ✅ 修复：expect_tx = kABytesTotal + kBBytes（与普通 PingPong 完全相同）
```

#### 典型二：跨 CTA mbarrier 竞争（Leader-Follower 模式）

两个 CTA 同时设置对方 mbarrier 的 `expect_tx` 存在竞争。最终方案：CTA0 作为 Leader 统一设置两个 CTA 的 `expect_tx`，使用 `mapa.shared::cluster` 跨 CTA 地址映射原语：

```cpp
// detail/mbarrier.cuh 新增跨 CTA 原语
CUTE_DEVICE void mbar_arrive_and_expect_tx_remote(
    uint64_t* mbar, int tx_bytes, int cta_id, int pred = 1)
{
    uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(mbar);
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  .reg .b32 remAddr32;\n\t"
        "  setp.eq.u32 p, %2, 1;\n\t"
        "  @p mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"  // 映射远端地址
        "  @p mbarrier.arrive.expect_tx.shared::cluster.b64 _, [remAddr32], %3;\n\t"
        "}"
        :: "r"(smem_ptr), "r"(cta_id), "r"(pred), "r"(tx_bytes) : "memory"
    );
}

// kernel 中：CTA0 作为 Leader 统一设置
if (cta_rank == 0) {
    mbar_arrive_and_expect_tx(&mbar_full[s], kExpectTx);           // 设置本地 CTA0 mbar
    mbar_arrive_and_expect_tx_remote(&mbar_full[s], kExpectTx, 1); // 设置 CTA1 mbar
}
```

#### `cudaLaunchKernelEx` 调用方式

```cpp
cudaLaunchConfig_t cfg = {};
cfg.gridDim  = grid;
cfg.blockDim = block;
cfg.dynamicSmemBytes = smem_size;
cudaLaunchAttribute attr[1];
attr[0].id = cudaLaunchAttributeClusterDimension;
attr[0].val.clusterDim = {2, 1, 1};
cfg.numAttrs = 1; cfg.attrs = attr;
// ✅ 模板版本，直接传 kernel 函数指针和参数（勿用 void** 数组形式）
CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel_func, arg1, arg2, ...));
```

### ⚠️ 环境限制（H20-3e + Driver 550.127.08）

**症状**：程序在 `cudaMalloc`（CUDA Context 初始化阶段）永久挂起。

**根因**：CUDA driver 加载含 `__cluster_dims__` 属性的 cubin 时死锁，与 kernel 逻辑无关。

**关键证据**：只要编译单元中包含含 `__cluster_dims__` 属性的 kernel **函数定义**（即使不调用），`cudaMalloc` 就会卡死。注释掉定义后恢复正常 → 确认卡死发生在 **CUDA driver 加载 cubin 阶段**。

**诊断方法**：
```bash
# 验证 cubin 内容正确
cuobjdump --dump-ptx main.o | grep -A3 "cluster"
# 若注释掉 cluster kernel 函数定义（不是调用），挂起消失 → 确认是驱动加载问题
cuobjdump --elf-section EIATTR main.o | grep EXPLICIT_CLUSTER  # 确认 cluster 属性已正确标记
```

**建议**：Cluster kernel 需要在以下环境验证：
- CUDA Driver ≥ 560（或确认 550 支持 cluster 的版本）
- 物理机（非虚拟化/容器）H100/H20 环境
- 确认 SM90a 的 cluster 调度权限配置

---

## 12. 性能调试工具与方法

### 寄存器分析

```bash
# 编译时查看 ptxas 统计
nvcc ... --ptxas-options=-v 2>&1 | grep "registers\|spill"

# 从已编译的 binary 查看
cuobjdump --dump-ptx main | grep ".reg .f32"   # 寄存器声明数
cuobjdump --dump-sass main | grep "^.*LDGSTS"  # 具体指令

# 指定寄存器上限（验证 spill 影响）
nvcc ... -maxrregcount=128
```

### nsys 非交互式 profiling

```bash
nsys profile --stats=true \
    --trace=cuda \
    -o profile_output \
    ./gemm_benchmark

# 查看 kernel 统计
nsys stats --report cuda_gpu_kern_sum profile_output.nsys-rep
```

### Occupancy 快速验证

```bash
# 通过 ptxas 输出计算
reg_per_thread=154; threads=160
echo "scale=2; 65536 / ($reg_per_thread * $threads)" | bc
# → 2.66 → floor = 2 blocks ✅

smem_per_block=98352; smem_per_sm=$((227*1024))
echo "scale=2; $smem_per_sm / $smem_per_block" | bc
# → 2.36 → floor = 2 blocks ✅
```

---

## 13. 经验结论速查表

### ✅ 必须做

| 规则 | 说明 |
|------|------|
| `GMMA::ss_op_selector<EA,EB,EC,TileShape,MajA,MajB>()` | 自动选择最优 MMA Op 类型，换 Tile/类型时无需手动查表 |
| `ss_smem_selector<Major,Elem,BLK_MN,BLK_K>()` | 自动选择最优 SMEM Swizzle（SW128/SW64/SW32/INTER）|
| `Swizzle<3,4,3>` | SM90 SMEM layout，M 参数必须是 4（对应 SW128，kTileK=64 时最优）|
| `TiledMMA` 不堆叠 AtomLayout | SM90_64xNx16 已是完整 WarpGroup MMA |
| Consumer 函数 `__forceinline__` | 避免 C7510，防止 wgmma pipeline 序列化 |
| Producer 函数 `__noinline__` + TMA 视图内建 | 寄存器隔离，防止污染 Consumer 路径 |
| `__launch_bounds__(160, 2)` | 引导 ptxas 将寄存器优化到 occupancy=2 |
| mbarrier 8B 对齐 | 动态 SMEM 最开头放 mbarrier 数组 |
| WAR 修复：每 tile 末尾 `arrive(mbar_empty)` | 缺少此行会导致结果错误或死锁 |
| `wgmma_wait<0>()` 在最后一个 tile | 确保 MMA 结果写回 tCrC |
| 申请超过 48KB SMEM | 用 `cudaFuncSetAttribute` 设置最大动态 SMEM |

### ❌ 常见陷阱

| 陷阱 | 后果 | 修复 |
|------|------|------|
| `expect_tx` 字节数错误 | mbar_full 永不触发，kernel 死锁 | 仔细计算每个 CTA 接收的 TMA 字节数 |
| Consumer `__noinline__` | C7510，wgmma 序列化，性能损失 15% | 改为 `__forceinline__` |
| TMA 视图在 kernel 本体构建后传入 Producer | 寄存器污染，232 reg → occupancy=1 | 将 TMA 视图移入 Producer 函数内部 |
| `setmaxnreg` 用于普通 kernel | 无效，occupancy 不提升 | 通过线程数 + `__launch_bounds__` 控制 |
| kStage=4（128KB SMEM） | occupancy=1，慢 12% | 用 kStage=3（96KB）|
| `__shared__` 静态变量 + 动态 SMEM mbarrier | mbarrier 对齐破坏，运行时 misaligned 错误 | 将辅助变量放到动态 SMEM 末尾 |
| AtomLayout 堆叠 TiledMMA | 需要 256 线程，结果全零 | 单 atom，无堆叠 |
| mbar_empty 未预 arrive | 流水线预热阶段 Producer 被 empty 屏障阻塞 | 初始化时 arrive kStage 次 |

### 性能数据参考（H20-3e, 4096×4096×2048, BF16→FP32）

| 版本 | 时间 | TFLOPS | vs cuBLAS |
|------|------|--------|-----------|
| cuBLAS baseline | 0.524 ms | 131.0 | 100% |
| TMA+GMMA（WAR 竞争版） | 1.146 ms | 60.0 | 45.7% |
| TMA+GMMA（WAR 修复后，kStage=3） | 0.524 ms | 131.1 | **100.7%** ✓ |
| PingPong v4（全 forceinline，232 reg） | 0.590 ms | 116.4 | 88.8% |
| **PingPong v5（寄存器隔离，154 reg）** | **0.524 ms** | **131.1** | **100.7%** ✅ |
| Persistent Kernel（156 blocks） | 0.527 ms | 130.4 | 99.5% |
| Cluster TMA Multicast | ❌ 驱动挂起（Driver 550 加载 cluster cubin 失败） | — | — |

---

## 14. 不同规模性能参考

### H20-3e 实测性能（PingPong v5, BF16→FP32）

| 规模 | cuBLAS | PingPong v5 | vs cuBLAS |
|------|--------|------------|--------|
| 2048×2048×1024 | 122.7 TFLOPS | 106.0 TFLOPS | 86.4% |
| 4096×4096×2048 | 131.0 TFLOPS | **131.1 TFLOPS** | **100.7%** ✅ |
| 8192×8192×4096 | 119.2 TFLOPS | **120.6 TFLOPS** | **101.2%** ✅ |

### 性能随规模变化的内在逻辑

**小规模性能下降原因**：

```
2048×2048×1024: (2048/128)×(2048/128) = 256 tiles
H20 occupancy=2 → 78×2 = 156 活跃 blocks
256 tiles / 156 blocks ≈ 1.6 waves → 第2个 wave 只有 100 blocks 有效
SM 平均利用率 ≈ 62%，性能约 106 TFLOPS
```

**大规模超越 cuBLAS 的原因**：

```
8192×8192×4096: (8192/128)×(8192/128) = 4096 tiles
4096 / 156 ≈ 26 waves → SM 长期满载
手写 Ping-Pong 流水线更充分，超越 cuBLAS ~1.2%
```

### Persistent Kernel 对不同规模的收益分析

```
小规模（256 tiles / 156 blocks ≈ 1.6 waves）：
  普通 kernel 第2个 wave 约 100 blocks，存在 56 个 SM 空闲
  Persistent 动态分配可均衡负载，小规模收益相对明显
  但 atomicAdd 开销依然存在，综合收益有限

中大规模（1024+ tiles）：
  Wave 切换开销 < 0.5%，Persistent 无明显优势
  实测: Persistent 130.4 TFLOPS vs 普通 131.1 TFLOPS（略慢 0.5%）
```

**结论**：Persistent Kernel 适合 tiles 数量 ≤ 2 waves 的小规模问题。对于 GEMM 中大规模，普通多-wave 调度 + work-stealing 的收益与开销相抵。

---

*整理自 SM90 GEMM 手写优化实践，2026-04-01（补充 Cluster TMA Multicast 调试经验、多规模性能数据、Persistent Kernel 定量结论）*
