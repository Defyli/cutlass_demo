#!/bin/bash
# ==============================================================================
# SM90 GEMM 性能分析脚本 (容器友好版, 不依赖 ncu 权限)
#
# 用法:
#   bash analysis.sh [选项]
#
# 选项:
#   --build          先重新编译再分析 (默认: 只分析)
#   --quick          只跑 nsys + cuobjdump, 跳过 clock64 (更快)
#   --m M --n N --k K  矩阵规模 (默认 4096 4096 2048)
#   --out DIR        报告输出目录 (默认 ./profile_reports)
#
# 分析内容:
#   [1] nsys profile   — kernel 时间线 + 占比统计
#   [2] cuobjdump      — 寄存器/SMEM 用量 + SASS 指令分布
#   [3] clock64 计时   — Consumer mbar_wait stall 周期 (device-side)
#   [4] 自动摘要报告   — 综合以上输出, 输出 summary.txt 供 AI 分析
# ==============================================================================
set -euo pipefail

# --------------------------------------------------------------------------
# 参数解析
# --------------------------------------------------------------------------
BUILD=0
QUICK=0
M=4096; N=4096; K=2048
OUT_DIR="./profile_reports"

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)  BUILD=1; shift ;;
        --quick)  QUICK=1; shift ;;
        --m) M=$2; shift 2 ;;
        --n) N=$2; shift 2 ;;
        --k) K=$2; shift 2 ;;
        --out) OUT_DIR=$2; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/build/gemm_sm90_demo"
mkdir -p "$OUT_DIR"

# --------------------------------------------------------------------------
# 工具检查
# --------------------------------------------------------------------------
echo "============================================================"
echo "  SM90 GEMM 性能分析  (M=$M N=$N K=$K)"
echo "  报告目录: $OUT_DIR"
echo "============================================================"

check_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo "[WARN] $1 not found, skipping related analysis."
        return 1
    fi
    return 0
}

HAS_NSYS=0;  check_tool nsys       && HAS_NSYS=1
HAS_CUOBJ=0; check_tool cuobjdump  && HAS_CUOBJ=1
HAS_NCU=0;   check_tool ncu        && HAS_NCU=1

# --------------------------------------------------------------------------
# Step 0: 可选编译
# --------------------------------------------------------------------------
if [[ $BUILD -eq 1 ]]; then
    echo ""
    echo "[0] 重新编译..."
    bash "${SCRIPT_DIR}/build.sh"
fi

if [[ ! -f "$BINARY" ]]; then
    echo "[ERROR] Binary not found: $BINARY"
    echo "        请先运行: bash build.sh  或使用 --build 参数"
    exit 1
fi

SUMMARY="${OUT_DIR}/summary.txt"
cat > "$SUMMARY" <<EOF
============================================================
  SM90 GEMM 性能分析报告
  时间: $(date)
  规模: M=$M N=$N K=$K
  二进制: $BINARY
============================================================

EOF

# --------------------------------------------------------------------------
# Step 1: 基准运行 — 记录各 kernel 原始性能
# --------------------------------------------------------------------------
echo ""
echo "[1] 基准性能测量..."
{
    echo "=== [1] 基准性能 (原始输出) ==="
    "$BINARY" "$M" "$N" "$K" 2>&1
    echo ""
} | tee "${OUT_DIR}/baseline.txt"
cat "${OUT_DIR}/baseline.txt" >> "$SUMMARY"

# --------------------------------------------------------------------------
# Step 2: nsys — kernel 时间线与统计
# --------------------------------------------------------------------------
if [[ $HAS_NSYS -eq 1 ]]; then
    echo ""
    echo "[2] nsys 分析..."

    NSYS_REP="${OUT_DIR}/gemm_sm90"
    NSYS_STATS="${OUT_DIR}/nsys_stats.txt"

    # 运行 nsys profile (--sample=none 减少开销, warmup 跑完后正式计时)
    nsys profile \
        --trace=cuda \
        --sample=none \
        --output "${NSYS_REP}" \
        --force-overwrite true \
        --stats false \
        "$BINARY" "$M" "$N" "$K" \
        2>&1 | grep -v "^Generating\|^Processing\|^SKIPPED" || true

    # 提取各类统计报告
    {
        echo "=== [2] nsys kernel 时间统计 (cuda_gpu_kern_sum) ==="
        nsys stats "${NSYS_REP}.nsys-rep" --report cuda_gpu_kern_sum 2>/dev/null || \
        nsys stats "${NSYS_REP}.nsys-rep" --report gpukernsum 2>/dev/null || \
        echo "(nsys stats 不可用或格式不同)"
        echo ""

        echo "=== [2] nsys CUDA API 统计 (cuda_api_sum) ==="
        nsys stats "${NSYS_REP}.nsys-rep" --report cuda_api_sum 2>/dev/null || \
        nsys stats "${NSYS_REP}.nsys-rep" --report cudaapisum 2>/dev/null || \
        echo "(cuda_api_sum 不可用)"
        echo ""
    } | tee "$NSYS_STATS"

    cat "$NSYS_STATS" >> "$SUMMARY"
    echo "    → 报告: ${NSYS_REP}.nsys-rep"
fi

# --------------------------------------------------------------------------
# Step 3: cuobjdump — 寄存器/SMEM + SASS 指令分布
# --------------------------------------------------------------------------
if [[ $HAS_CUOBJ -eq 1 ]]; then
    echo ""
    echo "[3] cuobjdump 分析..."

    CUOBJ_OUT="${OUT_DIR}/cuobjdump.txt"
    {
        echo "=== [3a] 资源用量 (寄存器数 / SMEM 字节) ==="
        cuobjdump --res-usage "$BINARY" 2>/dev/null
        echo ""

        echo "=== [3b] Ping-Pong kernel SASS 指令分布 ==="
        echo "--- WGMMA / MBAR / LDG / STG 指令统计 ---"
        cuobjdump -sass "$BINARY" 2>/dev/null \
          | awk '
              /\.entry |\.function / { cur = $0 }
              /WGMMA/  { wgmma[cur]++ }
              /MBAR/   { mbar[cur]++ }
              /LDG/    { ldg[cur]++ }
              /STG/    { stg[cur]++ }
              /LDSM/   { ldsm[cur]++ }
              END {
                for (k in wgmma) {
                  printf "Function: %s\n  WGMMA=%d  MBAR=%d  LDG=%d  STG=%d  LDSM=%d\n",
                    k, wgmma[k], mbar[k]+0, ldg[k]+0, stg[k]+0, ldsm[k]+0
                }
              }
            ' || echo "(SASS 解析失败)"
        echo ""

        echo "=== [3c] gemm_kernel_pingpong SASS 关键段 (前200行) ==="
        cuobjdump -sass "$BINARY" 2>/dev/null \
          | awk '/gemm_kernel_pingpong/{p=1;c=0} p{print; c++; if(c>200) p=0}' \
          || echo "(未找到 gemm_kernel_pingpong)"
        echo ""

        echo "=== [3d] gemm_kernel_tma SASS 关键段 (前200行) ==="
        cuobjdump -sass "$BINARY" 2>/dev/null \
          | awk '/gemm_kernel_tma</{p=1;c=0} p{print; c++; if(c>200) p=0}' \
          || echo "(未找到 gemm_kernel_tma)"
        echo ""
    } | tee "$CUOBJ_OUT"

    cat "$CUOBJ_OUT" >> "$SUMMARY"
fi

# --------------------------------------------------------------------------
# Step 4: device-side clock64 stall 分析
# (在 kernel 里插入计时点, 单独编译一个 debug build)
# --------------------------------------------------------------------------
if [[ $QUICK -eq 0 ]]; then
    echo ""
    echo "[4] clock64 stall 分析..."

    STALL_SRC="${OUT_DIR}/gemm_stall_probe.cu"
    STALL_BIN="${OUT_DIR}/gemm_stall_probe"

    # 生成临时探针源文件
    cat > "$STALL_SRC" <<'CUDA_EOF'
// 临时 stall 探针: 测量 Ping-Pong Consumer mbar_wait 实际 stall 周期
// 由 analysis.sh 自动生成, 勿手动修改
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>

// 复用主项目的 SMEM 参数 (128x128x64 BF16, kStage=4)
static constexpr int kTileM  = 128;
static constexpr int kTileN  = 128;
static constexpr int kTileK  = 64;
static constexpr int kStage  = 4;
static constexpr int kBytes  = kTileM * kTileK * 2 + kTileN * kTileK * 2; // BF16

struct alignas(8) MBarrier { uint64_t v; };

__device__ void mbar_init(MBarrier* b, int cnt) {
    uint32_t p = __cvta_generic_to_shared(b);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(p), "r"(cnt));
}
__device__ void mbar_arrive_tx(MBarrier* b, int tx) {
    uint32_t p = __cvta_generic_to_shared(b);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(p), "r"(tx));
}
__device__ void mbar_arrive(MBarrier* b) {
    uint32_t p = __cvta_generic_to_shared(b);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(p));
}
__device__ void mbar_wait(MBarrier* b, int phase) {
    uint32_t p = __cvta_generic_to_shared(b);
    asm volatile(
        "{\n.reg .pred done;\n"
        "LOOP%=: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1;\n"
        "@done bra DONE%=;\n"
        "nanosleep.u32 0x100;\n"
        "bra LOOP%=;\n"
        "DONE%=: }\n"
        :: "r"(p), "r"(phase) : "memory");
}

// stall 统计存储 (device)
__device__ long long g_stall_cycles[kStage * 2]; // [full_wait, empty_wait] * kStage
__device__ int       g_stall_count;

__global__ void stall_probe_kernel(int num_k_tiles)
{
    extern __shared__ char smem[];
    MBarrier* mbar_full  = (MBarrier*)(smem);
    MBarrier* mbar_empty = mbar_full + kStage;
    char*     fake_data  = (char*)(mbar_empty + kStage);

    int tid    = threadIdx.x;
    int wg_tid = tid % 128;
    bool is_producer = (tid >= 128);

    // 初始化 mbarrier
    if (tid == 0) {
        asm volatile("fence.mbarrier_init.release.cluster;\n" :::);
        for (int s = 0; s < kStage; s++) {
            mbar_init(&mbar_full[s],  1);
            mbar_init(&mbar_empty[s], 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" :::);
    }
    __syncthreads();
    // 预 arrive mbar_empty (所有 stage 初始空闲)
    if (!is_producer && wg_tid == 0)
        for (int s = 0; s < kStage; s++) mbar_arrive(&mbar_empty[s]);
    __syncthreads();

    long long total_full_stall  = 0;
    long long total_empty_stall = 0;
    int stage = 0, phase_f = 0, phase_e = 0;

    if (is_producer) {
        if (wg_tid == 0) {
            for (int k = 0; k < num_k_tiles; k++) {
                long long t0 = clock64();
                mbar_wait(&mbar_empty[stage], phase_e);
                long long t1 = clock64();
                total_empty_stall += (t1 - t0);

                // 模拟 TMA 完成: 直接 arrive_tx (无实际数据搬运)
                mbar_arrive_tx(&mbar_full[stage], kBytes);
                // 假装 TMA 立即完成 (再 arrive 一次把 tx_count 清零)
                // 实际 TMA 由硬件 complete_tx, 这里我们用 nanosleep 模拟延迟
                asm volatile("nanosleep.u32 500;\n" :::); // 模拟~500ns TMA延迟

                stage = (stage + 1) % kStage;
                if (stage == 0) phase_e ^= 1;
            }
            // 写到 gmem
            atomicAdd((unsigned long long*)g_stall_cycles + kStage,
                      (unsigned long long)total_empty_stall);
        }
    } else {
        int prev_stage = -1;
        for (int k = 0; k < num_k_tiles; k++) {
            long long t0 = clock64();
            mbar_wait(&mbar_full[stage], phase_f);
            long long t1 = clock64();
            total_full_stall += (t1 - t0);

            // 模拟 wgmma (用 nanosleep 模拟计算时间)
            asm volatile("nanosleep.u32 200;\n" :::);

            if (prev_stage >= 0 && wg_tid == 0)
                mbar_arrive(&mbar_empty[prev_stage]);

            prev_stage = stage;
            stage = (stage + 1) % kStage;
            if (stage == 0) phase_f ^= 1;
        }
        // drain
        if (prev_stage >= 0 && wg_tid == 0)
            mbar_arrive(&mbar_empty[prev_stage]);

        if (wg_tid == 0) {
            atomicAdd((unsigned long long*)g_stall_cycles,
                      (unsigned long long)total_full_stall);
            atomicAdd(&g_stall_count, num_k_tiles);
        }
    }
}

int main() {
    int num_k_tiles = 2048 / kTileK;  // K=2048, kTileK=64 -> 32 tiles

    // 重置 device 统计
    long long zero[kStage * 2] = {};
    int zero_cnt = 0;
    cudaMemcpyToSymbol(g_stall_cycles, zero, sizeof(zero));
    cudaMemcpyToSymbol(g_stall_count, &zero_cnt, sizeof(zero_cnt));

    // SMEM: mbar_full[4] + mbar_empty[4] + fake_data
    size_t smem = kStage * 2 * sizeof(MBarrier) + kBytes;

    // 暖机
    stall_probe_kernel<<<dim3(1), dim3(256), smem>>>(num_k_tiles);
    cudaDeviceSynchronize();

    // 重置后正式运行
    cudaMemcpyToSymbol(g_stall_cycles, zero, sizeof(zero));
    cudaMemcpyToSymbol(g_stall_count, &zero_cnt, sizeof(zero_cnt));
    stall_probe_kernel<<<dim3(4,4), dim3(256), smem>>>(num_k_tiles);
    cudaDeviceSynchronize();

    // 读取结果
    long long h_cycles[kStage * 2];
    int h_cnt;
    cudaMemcpyFromSymbol(h_cycles,    g_stall_cycles, sizeof(h_cycles));
    cudaMemcpyFromSymbol(&h_cnt,      g_stall_count,  sizeof(h_cnt));

    // 每个 block 只有 1 个 wg_tid==0 贡献
    // total blocks = 4*4 = 16
    int blocks = 16;
    long long avg_full  = h_cnt > 0 ? h_cycles[0] / h_cnt : 0;
    long long avg_empty = h_cnt > 0 ? h_cycles[kStage] / h_cnt : 0;
    // h_cycles[0] 是 16 个 block 累加, 除以 blocks 得到单 block 平均
    avg_full  = h_cycles[0] / blocks / num_k_tiles;
    avg_empty = h_cycles[kStage] / blocks / num_k_tiles;

    printf("\n=== clock64 stall 分析 (模拟 Ping-Pong 延迟) ===\n");
    printf("  num_k_tiles = %d  (K=2048, kTileK=64)\n", num_k_tiles);
    printf("  Consumer mbar_full  wait: 平均 %lld cycles / tile\n", avg_full);
    printf("  Producer mbar_empty wait: 平均 %lld cycles / tile\n", avg_empty);
    printf("\n  解读:\n");
    printf("    - full  wait > 1000 cycles => Consumer 等 Producer, TMA 是瓶颈\n");
    printf("    - full  wait ~  0   cycles => Consumer 不等,  wgmma 是瓶颈\n");
    printf("    - empty wait > 1000 cycles => Producer 等 Consumer, wgmma 过慢\n");
    printf("    - empty wait ~  0   cycles => Producer 不等, 流水线完全重叠\n");
    printf("\n  注意: 本探针用 nanosleep 模拟 TMA 和 wgmma 延迟,\n");
    printf("        实际结果以真实 kernel + nsys timeline 为准\n");

    return 0;
}
CUDA_EOF

    # 编译探针
    NVCC=$(command -v nvcc 2>/dev/null || echo "")
    if [[ -n "$NVCC" ]]; then
        CUTLASS_INC="${SCRIPT_DIR}/../../cutlass/include"
        "$NVCC" -arch=sm_90a -O2 \
            --expt-relaxed-constexpr \
            --generate-line-info \
            -I"${CUTLASS_INC}" \
            "$STALL_SRC" -o "$STALL_BIN" -lcudart 2>&1 \
            | tee "${OUT_DIR}/stall_probe_build.log" || {
                echo "    [WARN] 探针编译失败, 跳过 clock64 分析"
                echo "    → 查看: ${OUT_DIR}/stall_probe_build.log"
                STALL_BIN=""
            }
    else
        echo "    [WARN] nvcc not found, 跳过 clock64 分析"
        STALL_BIN=""
    fi

    if [[ -n "$STALL_BIN" && -f "$STALL_BIN" ]]; then
        {
            echo "=== [4] clock64 stall 分析 ==="
            "$STALL_BIN" 2>&1
            echo ""
        } | tee "${OUT_DIR}/stall_probe.txt"
        cat "${OUT_DIR}/stall_probe.txt" >> "$SUMMARY"
    fi
fi

# --------------------------------------------------------------------------
# Step 5: 汇总摘要 — 写入 AI 分析提示
# --------------------------------------------------------------------------
cat >> "$SUMMARY" <<'EOF'

============================================================
  AI 分析指引 (给 CatPaw)
============================================================

请根据以上报告回答以下问题:

1. [nsys 时间线] Ping-Pong kernel 与单WG TMA kernel 的 GPU 时间是否吻合?
   是否存在 CPU 侧 sync gap 或 kernel launch 开销?

2. [cuobjdump 资源] 两个 kernel 的寄存器数量差距多少?
   若 Ping-Pong REG > 单WG REG, occupancy 如何受影响?

3. [cuobjdump SASS] Ping-Pong Consumer 主循环中 WGMMA 指令密度如何?
   MBAR 指令数量是否已降至 ~num_k_tiles 级别?

4. [clock64] Consumer mbar_full wait 平均 stall 是多少 cycles?
   - 若 > 1000 cycles: TMA 延迟未被掩盖, 考虑增大 kStage 或更换策略
   - 若 ~0: wgmma 本身是瓶颈, 考虑寄存器/occupancy 优化

5. 综合以上, Ping-Pong 比单WG 慢 ~12% 的根本原因是什么?
   推荐的下一步优化方向?

EOF

echo ""
echo "============================================================"
echo "  分析完成!"
echo "  综合报告: $SUMMARY"
echo ""
echo "  各子报告:"
[[ -f "${OUT_DIR}/baseline.txt"     ]] && echo "    ${OUT_DIR}/baseline.txt"
[[ -f "${OUT_DIR}/nsys_stats.txt"   ]] && echo "    ${OUT_DIR}/nsys_stats.txt"
[[ -f "${NSYS_REP:-x}.nsys-rep" 2>/dev/null ]] && echo "    ${NSYS_REP}.nsys-rep"
[[ -f "${OUT_DIR}/cuobjdump.txt"    ]] && echo "    ${OUT_DIR}/cuobjdump.txt"
[[ -f "${OUT_DIR}/stall_probe.txt"  ]] && echo "    ${OUT_DIR}/stall_probe.txt"
echo "============================================================"
echo ""
echo "  下一步: 把 $SUMMARY 的内容粘贴给 AI, 即可获得针对性优化建议"
echo ""
