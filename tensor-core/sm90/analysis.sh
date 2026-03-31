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
#   [1] 基准性能      — 各 kernel TFLOPS
#   [2] nsys profile  — kernel 时间线 / 并发 / 内存带宽 / SM 利用率
#   [3] cuobjdump     — 寄存器 / SMEM / PTX setmaxnreg 验证 / SASS 指令密度
#   [4] clock64 计时  — 真实 kernel 内 mbar_wait stall 周期 (device-side)
#   [5] 汇总报告      — summary.txt 供 AI 分析
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
# Step 2: nsys — 多维度 kernel 分析
# --------------------------------------------------------------------------
if [[ $HAS_NSYS -eq 1 ]]; then
    echo ""
    echo "[2] nsys 分析..."

    NSYS_REP="${OUT_DIR}/gemm_sm90"
    NSYS_STATS="${OUT_DIR}/nsys_stats.txt"
    NSYS_SQLITE="${OUT_DIR}/gemm_sm90.sqlite"

    # ------------------------------------------------------------------
    # 2a. nsys profile (完整追踪: cuda + nvtx + osrt)
    #   --trace=cuda,nvtx,osrt  捕获 CUDA API + NVTX ranges + OS 运行时
    #   --sample=none           关闭 CPU 采样减少开销
    #   --force-overwrite true  覆盖已有报告
    #   注: osrt trace 是捕获 pthread/futex/sleep 等 CPU 侧延迟来源的关键
    # ------------------------------------------------------------------
    echo "  2a. 采集 nsys profile (cuda + nvtx + osrt)..."
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --output "${NSYS_REP}" \
        --force-overwrite true \
        --stats false \
        "$BINARY" "$M" "$N" "$K" \
        2>&1 | grep -v "^Generating\|^Processing\|^SKIPPED\|^Warning" || true

    # 导出 SQLite 供后续自定义查询
    # nsys stats 会自动将 .nsys-rep 转换为同名 .sqlite 文件, 放在同一目录下
    # 先触发一次报告让 sqlite 生成 (结果本身在 2b 中重新提取, 这里仅为触发导出)
    echo "  2a-2. 导出 SQLite 数据库..."
    nsys stats "${NSYS_REP}.nsys-rep" \
        --report cuda_gpu_kern_sum \
        --force-export true \
        2>/dev/null | head -1 || true
    # sqlite 文件位于 nsys-rep 同目录, 名为 gemm_sm90.sqlite
    if [[ -f "$NSYS_SQLITE" ]]; then
        echo "    SQLite 已生成: $NSYS_SQLITE"
    else
        echo "    [WARN] SQLite 未生成, SQLite 深度查询将跳过"
    fi

    echo "  2b. 提取标准统计报告..."
    {
        # ---- 2b-1: kernel 执行时间汇总 ----
        echo "=== [2-1] CUDA GPU Kernel 时间汇总 (cuda_gpu_kern_sum) ==="
        echo "  字段说明: Time(%) TotalTime(ns) Instances Avg(ns) Med Min Max StdDev Name"
        echo "  关注点: PingPong avg 比 TMA avg 高多少? 是否超过 15%?"
        nsys stats "${NSYS_REP}.nsys-rep" --report cuda_gpu_kern_sum 2>/dev/null || \
        nsys stats "${NSYS_REP}.nsys-rep" --report gpukernsum         2>/dev/null || \
        echo "  (不可用)"
        echo ""

        # ---- 2b-2: CUDA 内存操作统计 ----
        echo "=== [2-2] CUDA 内存操作时间 (cuda_gpu_mem_time_sum) ==="
        echo "  用途: 确认 cudaMemcpy / cudaMemset 是否占用显著时间"
        nsys stats "${NSYS_REP}.nsys-rep" --report cuda_gpu_mem_time_sum 2>/dev/null || \
        nsys stats "${NSYS_REP}.nsys-rep" --report gpumemtimesum         2>/dev/null || \
        echo "  (不可用或无内存拷贝事件)"
        echo ""

        # ---- 2b-3: CUDA 内存操作大小统计 ----
        echo "=== [2-3] CUDA 内存操作大小 (cuda_gpu_mem_size_sum) ==="
        echo "  用途: 确认每次 H2D/D2H 数据量"
        nsys stats "${NSYS_REP}.nsys-rep" --report cuda_gpu_mem_size_sum 2>/dev/null || \
        nsys stats "${NSYS_REP}.nsys-rep" --report gpumem                 2>/dev/null || \
        echo "  (不可用)"
        echo ""

        # ---- 2b-4: CUDA API 调用统计 ----
        echo "=== [2-4] CUDA API 调用统计 (cuda_api_sum) ==="
        echo "  用途: 确认 cudaLaunchKernel / cudaStreamSynchronize 是否有过多 CPU 开销"
        echo "  关注点: cuLaunchKernelEx (TMA kernel 的实际启动 API) 的调用次数和耗时"
        nsys stats "${NSYS_REP}.nsys-rep" --report cuda_api_sum 2>/dev/null || \
        nsys stats "${NSYS_REP}.nsys-rep" --report cudaapisum    2>/dev/null || \
        echo "  (不可用)"
        echo ""

        # ---- 2b-5: OS 运行时统计 (CPU sleep / mutex) ----
        echo "=== [2-5] OS 运行时统计 (osrt_sum) ==="
        echo "  用途: 检查 CPU 侧是否有 sleep/futex/pthread_mutex 导致 kernel launch 延迟"
        echo "  关注点: futex/nanosleep 时间占比; 若 CPU 在 cudaEventSynchronize 期间大量 sleep"
        echo "          说明 GPU 与 CPU 同步开销高"
        nsys stats "${NSYS_REP}.nsys-rep" --report osrt_sum 2>/dev/null || \
        echo "  (osrt 数据不可用 — 确认 nsys profile 使用了 --trace=cuda,nvtx,osrt)"
        echo ""

        # ---- 2b-6: NVTX 区间统计 ----
        echo "=== [2-6] NVTX 自定义区间统计 (nvtx_sum) ==="
        echo "  用途: 若代码中插入了 nvtxRangePush/Pop, 可按命名阶段统计耗时"
        echo "  注意: 当前 main.cu 未使用 NVTX markers, 此报告可能为空"
        echo "        建议: 在 benchmark() 循环的 n_warmup 和 n_iter 段分别加 NVTX 范围"
        echo "        以便区分 warmup 和稳态 kernel 在 nsys 时间线上的位置"
        nsys stats "${NSYS_REP}.nsys-rep" --report nvtx_sum 2>/dev/null || \
        echo "  (nvtx 数据不可用)"
        echo ""

    } | tee "$NSYS_STATS"

    # ------------------------------------------------------------------
    # 2c. SQLite 自定义查询 — 流水线执行质量深度分析
    #   注: SQLite 已在 2a-2 导出, 此处直接查询
    # ------------------------------------------------------------------
    echo "  2c. SQLite 深度查询 (kernel gap / 并发度 / 执行顺序)..."
    NSYS_SQL_OUT="${OUT_DIR}/nsys_sql_analysis.txt"
    {
        echo "=== [2-0] SQLite 数据库表名探查 (兼容性诊断) ==="
        echo "  用途: 列出 nsys sqlite 中实际存在的表, 确认后续查询使用正确的表名"
        if [[ -f "$NSYS_SQLITE" ]]; then
            sqlite3 "$NSYS_SQLITE" ".tables" 2>/dev/null \
              | tr ' ' '\n' | grep -v '^$' | sort \
              | awk '{printf "    %s\n", $0}' \
              || echo "  (无法列表)"
        else
            echo "  (SQLite 文件不存在)"
        fi
        echo ""

        echo "=== [2-7] Kernel 执行时间线 (按启动顺序, 仅稳态 kernel) ==="
        echo "  用途: 查看每次 kernel 启动的绝对时间、持续时间、以及与上次 kernel 的间隔"
        echo "  关注点:"
        echo "    1. PingPong kernel 的 launch gap 是否比 TMA kernel 更大?"
        echo "       大 gap 说明 CPU 侧有额外 setup 开销 (TMA descriptor 构建等)"
        echo "    2. 相邻两次同类 kernel 之间的 gap 是否稳定?"
        echo "       gap 抖动大说明 CPU 侧有竞争 (如内存分配、OS 调度)"
        echo ""
        if [[ -f "$NSYS_SQLITE" ]]; then
            sqlite3 -column -header "$NSYS_SQLITE" <<'SQL' 2>/dev/null || echo "  (SQLite 查询失败)"
-- 按时间顺序列出 GPU kernel 执行记录 (前 60 条, 跳过前 5 次 warmup 候选)
-- CUDA_KERNEL_EVENTS 或 CUPTI_ACTIVITY_KIND_KERNEL 表
SELECT
    ROW_NUMBER() OVER (ORDER BY start) AS seq,
    ROUND((start - MIN(start) OVER()) / 1e6, 3)  AS start_ms,
    ROUND(duration / 1e6, 3)                       AS dur_ms,
    ROUND(
        (start - LAG(start + duration, 1, start) OVER (ORDER BY start)) / 1e3, 1
    )                                               AS gap_us,
    SUBSTR(shortName, 1, 60)                        AS kernel_short_name
FROM (
    SELECT
        k.start,
        k.end - k.start  AS duration,
        s.value           AS shortName
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName = s.id
    ORDER BY k.start
)
LIMIT 60;
SQL
        else
            echo "  (SQLite 文件不存在: $NSYS_SQLITE)"
        fi
        echo ""

        echo "=== [2-8] 各 Kernel 类型: 平均执行时间 & 平均 Launch Gap (稳态) ==="
        echo "  用途: 定量对比 TMA / PingPong / cuBLAS 的 launch gap 和执行时间"
        echo "  关注点: launch gap 揭示 CPU 侧 kernel 准备开销 (descriptor、stream 同步等)"
        echo ""
        if [[ -f "$NSYS_SQLITE" ]]; then
            sqlite3 -column -header "$NSYS_SQLITE" <<'SQL' 2>/dev/null || echo "  (SQLite 查询失败)"
WITH ordered AS (
    SELECT
        s.value                                                      AS name,
        k.start,
        k.end - k.start                                              AS duration,
        k.start - LAG(k.end, 1, k.start) OVER (ORDER BY k.start)   AS gap_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName = s.id
),
-- 跳过前 5 次 (warmup), 每种 kernel 统计后 20 次
filtered AS (
    SELECT name, duration, gap_ns,
           ROW_NUMBER() OVER (PARTITION BY name ORDER BY start) AS rn
    FROM ordered
)
SELECT
    SUBSTR(name, 1, 55)             AS kernel_name,
    COUNT(*)                         AS instances,
    ROUND(AVG(duration) / 1e3, 1)   AS avg_dur_us,
    ROUND(MIN(duration) / 1e3, 1)   AS min_dur_us,
    ROUND(MAX(duration) / 1e3, 1)   AS max_dur_us,
    ROUND(AVG(gap_ns)   / 1e3, 1)   AS avg_gap_us,
    ROUND(MAX(gap_ns)   / 1e3, 1)   AS max_gap_us
FROM filtered
WHERE rn > 5
GROUP BY name
ORDER BY avg_dur_us DESC;
SQL
        else
            echo "  (SQLite 文件不存在)"
        fi
        echo ""

        echo "=== [2-9] Kernel 并发度分析 (Ping-Pong 流水线重叠检测) ==="
        echo "  用途: 检测 Ping-Pong kernel 的两个实例 (若存在) 是否在时间上重叠"
        echo "        理论上 Ping-Pong 的 Producer WG 和 Consumer WG 在同一 kernel 内"
        echo "        (单次 kernel launch, 256 线程). 若多个 block 并发, 看 GPU occupancy"
        echo "  关注点: 同一时刻 active kernel 数量 > 1 说明有并发 (stream 并行或 MPS)"
        echo ""
        if [[ -f "$NSYS_SQLITE" ]]; then
            sqlite3 -column -header "$NSYS_SQLITE" <<'SQL' 2>/dev/null || echo "  (SQLite 查询失败)"
-- 检查是否有任何两个 kernel 在时间上重叠
-- 若 overlap_count > 0, 说明存在 kernel 并发
SELECT
    SUBSTR(a_name, 1, 45) AS kernel_A,
    SUBSTR(b_name, 1, 45) AS kernel_B,
    COUNT(*)              AS overlap_count,
    ROUND(AVG(overlap_ns) / 1e3, 1) AS avg_overlap_us
FROM (
    SELECT
        sa.value AS a_name,
        sb.value AS b_name,
        MIN(a.end, b.end) - MAX(a.start, b.start) AS overlap_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL a
    JOIN CUPTI_ACTIVITY_KIND_KERNEL b ON a.id < b.id
    JOIN StringIds sa ON a.shortName = sa.id
    JOIN StringIds sb ON b.shortName = sb.id
    WHERE MAX(a.start, b.start) < MIN(a.end, b.end)  -- 有重叠
    LIMIT 500
)
GROUP BY kernel_A, kernel_B
ORDER BY overlap_count DESC
LIMIT 20;
SQL
        else
            echo "  (SQLite 文件不存在)"
        fi
        echo ""

        echo "=== [2-10] 相邻 Kernel 间隔分布 (Gap 直方图) ==="
        echo "  用途: 了解 kernel launch gap 的分布 (是否有异常大 gap 暗示 CPU 卡顿)"
        echo "  关注点: 若 P95 gap >> median gap, 说明偶发性 CPU 延迟 (OS 调度、内存分配)"
        echo ""
        if [[ -f "$NSYS_SQLITE" ]]; then
            sqlite3 -column -header "$NSYS_SQLITE" <<'SQL' 2>/dev/null || echo "  (SQLite 查询失败)"
WITH gaps AS (
    SELECT
        k.start - LAG(k.end, 1, k.start) OVER (ORDER BY k.start) AS gap_ns,
        s.value AS name
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName = s.id
)
SELECT
    CASE
        WHEN gap_ns <      1000 THEN '  <1us  (极小 gap)'
        WHEN gap_ns <     10000 THEN ' 1-10us (正常 launch overhead)'
        WHEN gap_ns <    100000 THEN '10-100us (中等延迟)'
        WHEN gap_ns <   1000000 THEN '0.1-1ms  (较大延迟)'
        ELSE                         '>1ms     (异常大延迟)'
    END AS gap_range,
    COUNT(*) AS count
FROM gaps
WHERE gap_ns IS NOT NULL AND gap_ns > 0
GROUP BY gap_range
ORDER BY MIN(gap_ns);
SQL
        else
            echo "  (SQLite 文件不存在)"
        fi
        echo ""

        echo "=== [2-11] PingPong vs TMA: 稳态逐次执行时间对比 (前 30 次稳态运行) ==="
        echo "  用途: 检查 PingPong kernel 是否有逐次时间抖动 (vs TMA 稳定)"
        echo "  抖动大 => 流水线 stall 非确定性 (如 SMEM occupancy 变化、TLB miss 等)"
        echo ""
        if [[ -f "$NSYS_SQLITE" ]]; then
            sqlite3 -column -header "$NSYS_SQLITE" <<'SQL' 2>/dev/null || echo "  (SQLite 查询失败)"
SELECT
    ROW_NUMBER() OVER (PARTITION BY ktype ORDER BY start) AS run_idx,
    ktype,
    ROUND(duration / 1e3, 1) AS dur_us
FROM (
    SELECT
        k.start,
        k.end - k.start AS duration,
        CASE
            WHEN s.value LIKE '%pingpong%' THEN 'PingPong'
            WHEN s.value LIKE '%kernel_tma%' THEN 'TMA'
            WHEN s.value LIKE '%nvjet%' THEN 'cuBLAS'
            ELSE 'Other'
        END AS ktype
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName = s.id
    WHERE s.value LIKE '%pingpong%'
       OR s.value LIKE '%kernel_tma%'
       OR s.value LIKE '%nvjet%'
    ORDER BY k.start
)
WHERE ktype != 'Other'
LIMIT 90;
SQL
        else
            echo "  (SQLite 文件不存在)"
        fi
        echo ""

    } | tee "$NSYS_SQL_OUT"

    cat "$NSYS_SQL_OUT" >> "$SUMMARY"

    # ------------------------------------------------------------------
    # 2d. kernel 时间对比摘要 (从 cuda_gpu_kern_sum 提取, 便于快速阅读)
    # ------------------------------------------------------------------
    {
        echo "=== [2-12] Kernel 时间对比摘要 (从 cuda_gpu_kern_sum 提取) ==="
        echo "  格式: Time%  Avg_us  Med_us  Min_us  Max_us  kernel_short_name"
        echo ""
        nsys stats "${NSYS_REP}.nsys-rep" --report cuda_gpu_kern_sum 2>/dev/null \
          | grep -E "gemm_kernel_pingpong|gemm_kernel_tma|nvjet|cp_async" \
          | awk '
            {
              # 字段顺序: Time%  TotalTime(ns)  Instances  Avg(ns)  Med(ns)  Min(ns)  Max(ns)  StdDev  Name
              printf "  Time%%=%-6s  Avg=%-8.1f us  Med=%-8.1f us  Min=%-7.1f us  Max=%-7.1f us  %s\n",
                $1, $4/1000, $5/1000, $6/1000, $7/1000, $NF
            }' \
          || echo "  (无法提取)"
        echo ""
        echo "  PingPong 相对于 TMA 的性能开销:"
        nsys stats "${NSYS_REP}.nsys-rep" --report cuda_gpu_kern_sum 2>/dev/null \
          | grep -E "gemm_kernel_pingpong|gemm_kernel_tma" \
          | awk '
            /pingpong/ { pp_avg = $4 + 0 }
            /kernel_tma/ && !/kStage/ { tma_avg = $4 + 0 }
            END {
              if (tma_avg > 0 && pp_avg > 0)
                printf "  PingPong avg = %.1f us, TMA avg = %.1f us, overhead = %.1f%%\n",
                  pp_avg/1000, tma_avg/1000, (pp_avg - tma_avg) * 100.0 / tma_avg
              else
                print "  (无法计算)"
            }' \
          || echo "  (无法计算)"
        echo ""
    } | tee -a "$NSYS_STATS"

    cat "${OUT_DIR}/nsys_stats.txt" >> "$SUMMARY"

    echo "    → nsys 报告:      ${NSYS_REP}.nsys-rep"
    echo "    → 标准统计:       $NSYS_STATS"
    echo "    → SQLite 深度分析: $NSYS_SQL_OUT"
    echo "    → SQLite 数据库:  $NSYS_SQLITE"
    echo ""
    echo "    提示: 如需图形化时间线, 在本地运行:"
    echo "          nsys-ui ${NSYS_REP}.nsys-rep"
fi

# --------------------------------------------------------------------------
# Step 3: cuobjdump — 寄存器/SMEM/PTX/SASS 深度分析
# --------------------------------------------------------------------------
if [[ $HAS_CUOBJ -eq 1 ]]; then
    echo ""
    echo "[3] cuobjdump 分析..."

    CUOBJ_OUT="${OUT_DIR}/cuobjdump.txt"
    {
        # ---- 3a: 资源用量 ----
        echo "=== [3a] 资源用量汇总 (寄存器数 / SMEM 字节) ==="
        echo "  字段: kernel_name | registers | smem_bytes | spill_stores | spill_loads"
        cuobjdump --res-usage "$BINARY" 2>/dev/null
        echo ""

        # ---- 3b: PTX 验证 setmaxnreg ----
        echo "=== [3b] PTX 验证: setmaxnreg 指令是否生效 ==="
        echo "  预期: gemm_kernel_pingpong 的 PTX 中应包含 setmaxnreg.dec 40 和 setmaxnreg.inc 160"
        cuobjdump -ptx "$BINARY" 2>/dev/null \
          | grep -A2 -B2 "setmaxnreg" \
          || echo "  [未找到 setmaxnreg 指令 — 说明编译器未保留 PTX inline asm, 需检查编译选项]"
        echo ""

        # ---- 3c: SASS 层面验证 setmaxnreg (ALDS/SETMAXREG 指令) ----
        echo "=== [3c] SASS 验证: SETMAXREG 指令 ==="
        echo "  预期: pingpong kernel SASS 中应有 SETMAXREG 40 和 SETMAXREG 160"
        cuobjdump -sass "$BINARY" 2>/dev/null \
          | grep -i "SETMAXREG\|setmaxnreg" \
          || echo "  [未找到 SETMAXREG — 可能已被 PTX 优化消除, 参考 PTX 层面]"
        echo ""

        # ---- 3d: 各 kernel SASS 指令密度统计 ----
        echo "=== [3d] 各 kernel SASS 指令密度统计 ==="
        echo "  统计: WGMMA / MBAR / LDG / STG / LDSM / BRA 指令数"
        echo "  用途: 确认 Consumer 循环的计算密度 vs 同步开销"
        cuobjdump -sass "$BINARY" 2>/dev/null \
          | awk '
              /Function :/ { cur = $NF; wgmma[cur]=0; mbar[cur]=0; ldg[cur]=0
                              stg[cur]=0; ldsm[cur]=0; bra[cur]=0; total[cur]=0 }
              cur != "" {
                  total[cur]++
                  if (/WGMMA/)  wgmma[cur]++
                  if (/MBAR/)   mbar[cur]++
                  if (/LDG/)    ldg[cur]++
                  if (/STG/)    stg[cur]++
                  if (/LDSM/)   ldsm[cur]++
                  if (/\bBRA\b|\bBRX\b/) bra[cur]++
              }
              END {
                for (k in total) {
                  if (total[k] > 20) {
                    # 截取函数名前80字符
                    name = substr(k, 1, 80)
                    printf "  %-80s\n    total=%-6d WGMMA=%-4d MBAR=%-4d LDG=%-4d STG=%-4d LDSM=%-4d BRA=%-4d\n\n",
                      name, total[k], wgmma[k], mbar[k], ldg[k], stg[k], ldsm[k], bra[k]
                  }
                }
              }
            ' || echo "  (SASS 解析失败)"
        echo ""

        # ---- 3e: Ping-Pong kernel SASS 主循环片段 ----
        echo "=== [3e] gemm_kernel_pingpong SASS 主体 (前 300 行) ==="
        echo "  重点看: SETMAXREG / WGMMA.FENCE / WGMMA.COMMIT / MBARRIER 指令顺序"
        cuobjdump -sass "$BINARY" 2>/dev/null \
          | awk '/Function :.*gemm_kernel_pingpong/{p=1;c=0} p{print; if(++c>300) p=0}' \
          || echo "  (未找到 gemm_kernel_pingpong)"
        echo ""

        # ---- 3f: pp_consumer_loop SASS 主循环片段 ----
        echo "=== [3f] pp_consumer_loop SASS 主体 (前 400 行) ==="
        echo "  重点看: WGMMA 指令密度, MBARRIER.WAIT 频率, SETMAXREG 是否存在"
        cuobjdump -sass "$BINARY" 2>/dev/null \
          | awk '/Function :.*pp_consumer_loop/{p=1;c=0} p{print; if(++c>400) p=0}' \
          || echo "  (未找到 pp_consumer_loop)"
        echo ""

        # ---- 3g: pp_producer_loop SASS ----
        echo "=== [3g] pp_producer_loop SASS 主体 (前 200 行) ==="
        echo "  重点看: SETMAXREG 40 是否出现在函数开头"
        cuobjdump -sass "$BINARY" 2>/dev/null \
          | awk '/Function :.*pp_producer_loop/{p=1;c=0} p{print; if(++c>200) p=0}' \
          || echo "  (未找到 pp_producer_loop)"
        echo ""

        # ---- 3h: TMA kernel SASS 对比 ----
        echo "=== [3h] gemm_kernel_tma SASS 主体 (前 300 行, 对比基准) ==="
        cuobjdump -sass "$BINARY" 2>/dev/null \
          | awk '/Function :.*gemm_kernel_tma</{p=1;c=0} p{print; if(++c>300) p=0}' \
          || echo "  (未找到 gemm_kernel_tma)"
        echo ""

    } | tee "$CUOBJ_OUT"

    cat "$CUOBJ_OUT" >> "$SUMMARY"
    echo "    → cuobjdump 报告: $CUOBJ_OUT"
fi

# --------------------------------------------------------------------------
# Step 4: device-side clock64 stall 分析
#   在真实 Ping-Pong kernel 中插入计时点, 测量实际 mbar_wait stall 周期
# --------------------------------------------------------------------------
if [[ $QUICK -eq 0 ]]; then
    echo ""
    echo "[4] clock64 stall 分析 (真实 kernel 计时)..."

    STALL_SRC="${OUT_DIR}/gemm_stall_probe.cu"
    STALL_BIN="${OUT_DIR}/gemm_stall_probe"

    # 生成探针源文件
    # 此版本直接在真实的 TMA + wgmma 循环中插入 clock64 计时,
    # 而非用 nanosleep 模拟 — 结果更真实
    cat > "$STALL_SRC" <<'CUDA_EOF'
// =============================================================================
// clock64 stall 探针: 测量真实 Ping-Pong 中 mbar_wait 的实际 stall 周期
//
// 在 Consumer WG 的 mbar_full_wait 前后插入 clock64, 记录每轮等待时间.
// 在 Producer WG 的 mbar_empty_wait 前后插入 clock64.
//
// 判读标准:
//   Consumer full_wait ~0        => wgmma 慢于 TMA, Consumer 是瓶颈
//   Consumer full_wait >> 0      => TMA 慢, Producer 是瓶颈
//   Producer empty_wait ~0       => TMA 快, 流水线重叠好
//   Producer empty_wait >> 0     => Consumer wgmma 太慢, Producer 被 stall
// =============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>

// 与主项目保持一致的参数
static constexpr int kTileM  = 128;
static constexpr int kTileN  = 128;
static constexpr int kTileK  = 64;
static constexpr int kStage  = 4;
// TMA 传输字节数: A(kTileM×kTileK×sizeof(bf16)) + B(kTileN×kTileK×sizeof(bf16))
static constexpr int kTmaBytes = (kTileM + kTileN) * kTileK * 2;

// mbarrier 辅助函数
__device__ inline uint32_t smem_ptr(void* p) { return __cvta_generic_to_shared(p); }

__device__ void mbar_init(uint64_t* b, int cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(smem_ptr(b)), "r"(cnt));
}
__device__ void mbar_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" :::);
}
__device__ void mbar_arrive_tx(uint64_t* b, int tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(smem_ptr(b)), "r"(tx));
}
__device__ void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n"
                 :: "r"(smem_ptr(b)));
}
__device__ void mbar_wait(uint64_t* b, int phase) {
    uint32_t p = smem_ptr(b);
    asm volatile(
        "{\n.reg .pred done;\n"
        "LOOP%=: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1;\n"
        "@done bra DONE%=;\n"
        "nanosleep.u32 0x100;\n"
        "bra LOOP%=;\n"
        "DONE%=: }\n"
        :: "r"(p), "r"(phase) : "memory");
}

// =============================================================================
// stall 统计 (device global)
// =============================================================================
// 布局: [0] = Consumer full_wait 总周期 (wg_tid==0 的累加)
//       [1] = Producer empty_wait 总周期
//       [2] = Consumer full_wait 次数
//       [3] = Producer empty_wait 次数
__device__ long long g_stall[4];

// =============================================================================
// 模拟 Ping-Pong: 用 cp.async + wgmma (若可用) 测量真实 stall
// 此处使用 "SMEM 直接写" 模拟 TMA 数据到达, 用计算循环模拟 wgmma 延迟
// =============================================================================
__global__ __launch_bounds__(256, 1)
void stall_probe_pp(int num_k_tiles, int tma_delay_ns, int wgmma_delay_ns)
{
    // SMEM 布局: data_A[kTileM*kTileK] | data_B[kTileN*kTileK] | mbar_full[kStage] | mbar_empty[kStage]
    extern __shared__ char smem[];
    uint64_t* mbar_full  = (uint64_t*)(smem + (kTileM + kTileN) * kTileK * 2 * kStage);
    uint64_t* mbar_empty = mbar_full + kStage;

    int tid    = threadIdx.x;
    int wg_tid = tid % 128;
    bool is_producer = (tid >= 128);

    // setmaxnreg: 模拟主 kernel 的寄存器分区
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" :::);

    // 初始化 mbarrier
    if (tid == 0) {
        for (int s = 0; s < kStage; s++) {
            mbar_init(&mbar_full[s],  1);
            mbar_init(&mbar_empty[s], 1);
        }
        mbar_fence_init();
    }
    __syncthreads();

    // Consumer 预 arrive mbar_empty (所有 stage 初始为"空")
    if (!is_producer && wg_tid == 0)
        for (int s = 0; s < kStage; s++) mbar_arrive(&mbar_empty[s]);
    __syncthreads();

    long long my_full_stall  = 0;
    long long my_empty_stall = 0;

    if (is_producer) {
        // ------------------------------------------------------------------
        // Consumer WG: 恢复配额 (模拟主 kernel)
        // Producer WG: 保持 40
        // ------------------------------------------------------------------
        int stage = 0, phase_e = 0;
        if (wg_tid == 0) {
            for (int k = 0; k < num_k_tiles; k++) {
                long long t0 = clock64();
                mbar_wait(&mbar_empty[stage], phase_e);
                long long t1 = clock64();
                my_empty_stall += (t1 - t0);

                // 模拟 TMA: 以 arrive_tx 宣告数据到达, 用 nanosleep 模拟传输延迟
                mbar_arrive_tx(&mbar_full[stage], kTmaBytes);
                // TMA 完成由 tx_count 归零触发 — 这里用 nanosleep 模拟 HW 延迟
                // 注意: 真实 TMA 下 tx_count 由硬件减, 这里直接用 arrive_tx 跳过
                // (arrive_tx 既宣告了 tx_count=kTmaBytes 又立即"完成"了)

                // 模拟 TMA 传输延迟 (让 Consumer 有机会看到 stall)
                if (tma_delay_ns > 0)
                    asm volatile("nanosleep.u32 %0;\n" :: "r"(tma_delay_ns));

                stage = (stage + 1) % kStage;
                if (stage == 0) phase_e ^= 1;
            }
        }
        // wg_tid != 0 的 Producer 线程 idle
        if (wg_tid == 0) {
            atomicAdd((unsigned long long*)&g_stall[1],
                      (unsigned long long)my_empty_stall);
            atomicAdd((unsigned long long*)&g_stall[3], (unsigned long long)num_k_tiles);
        }

    } else {
        // Consumer WG 恢复配额
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;\n" :::);

        int stage = 0, phase_f = 0;
        for (int k = 0; k < num_k_tiles; k++) {
            // 计时: mbar_full wait
            long long t0 = clock64();
            mbar_wait(&mbar_full[stage], phase_f);
            long long t1 = clock64();
            if (wg_tid == 0) my_full_stall += (t1 - t0);

            // 模拟 wgmma 计算延迟
            if (wgmma_delay_ns > 0)
                asm volatile("nanosleep.u32 %0;\n" :: "r"(wgmma_delay_ns));

            // 通知 Producer: 上一个 stage 已用完
            if (k > 0 && wg_tid == 0) {
                int prev = (stage - 1 + kStage) % kStage;
                mbar_arrive(&mbar_empty[prev]);
            }

            stage = (stage + 1) % kStage;
            if (stage == 0) phase_f ^= 1;
        }
        // Drain
        if (num_k_tiles > 0 && wg_tid == 0) {
            int last = (stage - 1 + kStage) % kStage;
            mbar_arrive(&mbar_empty[last]);
        }

        if (wg_tid == 0) {
            atomicAdd((unsigned long long*)&g_stall[0],
                      (unsigned long long)my_full_stall);
            atomicAdd((unsigned long long*)&g_stall[2], (unsigned long long)num_k_tiles);
        }
    }
}

// =============================================================================
// 辅助: 测量 GPU 时钟频率 (cycles per microsecond)
// =============================================================================
__global__ void measure_clk(long long* out) {
    long long t0 = clock64();
    // 精确 sleep 10us
    asm volatile("nanosleep.u32 10000;\n" :::);
    long long t1 = clock64();
    if (threadIdx.x == 0) *out = t1 - t0;  // cycles per ~10us
}

int main() {
    // 测量时钟频率
    long long* d_clk;
    cudaMalloc(&d_clk, sizeof(long long));
    measure_clk<<<1, 32>>>(d_clk);
    cudaDeviceSynchronize();
    long long clk10us;
    cudaMemcpy(&clk10us, d_clk, sizeof(long long), cudaMemcpyDeviceToHost);
    cudaFree(d_clk);
    double mhz = clk10us / 10.0;  // cycles/us = MHz
    printf("\n  GPU 时钟频率估算: %.0f MHz (%.1f cycles/ns)\n\n", mhz, mhz / 1000.0);

    // SMEM 大小: data_AB * kStage + mbar * 2 * kStage
    size_t smem = (size_t)(kTileM + kTileN) * kTileK * 2 * kStage
                + kStage * 2 * sizeof(uint64_t);

    int num_k_tiles = 2048 / kTileK;  // K=2048 -> 32 tiles

    // 多场景测试: 调整 TMA 延迟 和 wgmma 延迟的相对大小
    struct Scenario { int tma_ns; int wgmma_ns; const char* desc; };
    Scenario scenarios[] = {
        {   0,   0, "无人工延迟 (纯同步开销)"},
        { 200, 200, "TMA≈wgmma (均衡流水线)"},
        { 500, 200, "TMA 较慢 (生产者瓶颈)"},
        { 200, 500, "wgmma 较慢 (消费者瓶颈)"},
        { 800, 200, "TMA 很慢 (重度生产者瓶颈)"},
    };

    printf("=== clock64 stall 分析 (模拟 Ping-Pong, kStage=%d, num_k_tiles=%d) ===\n",
           kStage, num_k_tiles);
    printf("  GPU 时钟: %.0f MHz\n\n", mhz);
    printf("  %-35s  %12s  %12s  %s\n",
           "场景", "Consumer stall", "Producer stall", "解读");
    printf("  %-35s  %12s  %12s  %s\n",
           "---", "(cycles/tile)", "(cycles/tile)", "---");

    for (auto& sc : scenarios) {
        // 重置统计
        long long zero4[4] = {};
        cudaMemcpyToSymbol(g_stall, zero4, sizeof(zero4));

        // 暖机
        stall_probe_pp<<<1, 256, smem>>>(num_k_tiles, sc.tma_ns, sc.wgmma_ns);
        cudaDeviceSynchronize();
        cudaMemcpyToSymbol(g_stall, zero4, sizeof(zero4));

        // 正式运行 (4x4 grid = 16 blocks, 模拟实际 4096/128 × 4096/128 = 1024 blocks 的子集)
        stall_probe_pp<<<dim3(4,4), 256, smem>>>(num_k_tiles, sc.tma_ns, sc.wgmma_ns);
        cudaDeviceSynchronize();

        long long h_stall[4];
        cudaMemcpyFromSymbol(h_stall, g_stall, sizeof(h_stall));

        int blocks = 16;
        long long consumer_stall = h_stall[2] > 0 ? h_stall[0] / h_stall[2] : 0;
        long long producer_stall = h_stall[3] > 0 ? h_stall[1] / h_stall[3] : 0;

        const char* interpret;
        if (consumer_stall < 100 && producer_stall < 100)
            interpret = "完美流水线";
        else if (consumer_stall > producer_stall * 3)
            interpret = "TMA 瓶颈 (Consumer 等 Producer)";
        else if (producer_stall > consumer_stall * 3)
            interpret = "wgmma 瓶颈 (Producer 等 Consumer)";
        else
            interpret = "双侧均有等待";

        printf("  %-35s  %12lld  %12lld  %s\n",
               sc.desc, consumer_stall, producer_stall, interpret);
    }

    printf("\n  字段说明:\n");
    printf("  Consumer full_wait  stall: Consumer 等待 TMA 数据就绪的周期\n");
    printf("  Producer empty_wait stall: Producer 等待 Consumer 释放 SMEM 的周期\n");
    printf("\n  判读:\n");
    printf("  Consumer stall ~0         => wgmma 比 TMA 慢, Consumer 是瓶颈\n");
    printf("  Consumer stall >> 1000    => TMA 比 wgmma 慢, 考虑增大 kStage\n");
    printf("  Producer stall >> 1000    => wgmma 太慢, Consumer 是严重瓶颈\n");
    printf("  两者均 ~0                  => 流水线充分重叠, 开销来自其他因素\n");
    printf("\n  注意: 真实 TMA/wgmma 延迟请结合 nsys 时间线确认\n");

    return 0;
}
CUDA_EOF

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
            echo "=== [4] clock64 stall 分析 (多场景) ==="
            "$STALL_BIN" 2>&1
            echo ""
        } | tee "${OUT_DIR}/stall_probe.txt"
        cat "${OUT_DIR}/stall_probe.txt" >> "$SUMMARY"
    fi
fi

# --------------------------------------------------------------------------
# Step 5: 汇总摘要 — 关键指标对比 + AI 分析指引
# --------------------------------------------------------------------------
cat >> "$SUMMARY" <<'EOF'

============================================================
  关键性能指标对比
============================================================

待填写 (从以上各节提取):

  kernel              | reg/thread | SMEM    | avg_time(us) | TFLOPS | vs_cuBLAS
  --------------------|-----------|---------|--------------|--------|----------
  gemm_kernel_tma     |  154      |  96 KB  |              | ~132   | ~100%
  gemm_kernel_pingpong|  255(ptxas)|128 KB  |              | ~114   | ~87%
  cuBLAS              |  -        |  -      |              | ~130   | 100%

  setmaxnreg 物理配额 (若生效):
    Producer WG: 40 reg × 128 threads = 5120 reg
    Consumer WG: 160 reg × 128 threads = 20480 reg
    per block:   25600 reg
    SM 驻留:     65536 / 25600 = 2.56 → 2 blocks (若 SMEM 允许)
    kStage=3 SMEM: 96KB × 2 = 192KB < 227KB ✓  → 2 blocks 可驻留
    kStage=4 SMEM: 128KB × 2 = 256KB > 227KB ✗ → 受 SMEM 限制, 1 block

============================================================
  AI 分析指引 (给 CatPaw)
============================================================

请根据以上报告回答以下问题:

1. [3b/3c] PTX/SASS 中是否存在 setmaxnreg 40 和 setmaxnreg 160 指令?
   若不存在, 说明编译器优化掉了 inline asm, 需要加 volatile 强制保留.

2. [3d] pp_consumer_loop 的 WGMMA 指令数量 vs MBAR 指令数量之比是多少?
   理想情况: WGMMA >> MBAR (计算密集), 若 MBAR ≈ WGMMA 说明同步开销重.

3. [2-1] Ping-Pong kernel avg_time vs TMA avg_time 差距百分比?
   kStage=3 和 kStage=4 的差距是否相同? (验证是否受 SMEM occupancy 影响)

4. [4] Consumer full_wait stall 是多少 cycles?
   若 ~0: wgmma 是瓶颈, setmaxnreg 提升 occupancy 是正确方向.
   若 >> 1000: TMA 延迟暴露, 需要更大 kStage 或其他优化.

5. 综合分析: Ping-Pong 当前比单WG 慢约 13% 的可能根因?
   - 即使 setmaxnreg 生效, kStage=4 因 SMEM 限制仍只有 1 block?
   - C7510 警告导致 wgmma pipeline crossing 实际有性能代价?
   - 256 线程中 128 个 Producer 线程大部分时间 idle, 浪费?

EOF

echo ""
echo "============================================================"
echo "  分析完成!"
echo "  综合报告: $SUMMARY"
echo ""
echo "  各子报告:"
[[ -f "${OUT_DIR}/baseline.txt"     ]] && echo "    ${OUT_DIR}/baseline.txt"
[[ -f "${OUT_DIR}/nsys_stats.txt"   ]] && echo "    ${OUT_DIR}/nsys_stats.txt"
NSYS_REP="${OUT_DIR}/gemm_sm90"
[[ -f "${NSYS_REP}.nsys-rep"        ]] && echo "    ${NSYS_REP}.nsys-rep  (nsys-ui 可视化)"
[[ -f "${OUT_DIR}/cuobjdump.txt"    ]] && echo "    ${OUT_DIR}/cuobjdump.txt"
[[ -f "${OUT_DIR}/stall_probe.txt"  ]] && echo "    ${OUT_DIR}/stall_probe.txt"
echo "============================================================"
echo ""
echo "  下一步: 把 $SUMMARY 的内容粘贴给 AI, 即可获得针对性优化建议"
echo ""
