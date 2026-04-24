#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

// CuTe headers
#include "cute/tensor.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"

#include "gemm_fp8_sm90.cuh"

using namespace cute;

// ============================================================================
// 类型定义
// ============================================================================
using TA     = cutlass::float_e4m3_t;   // FP8 E4M3 输入
using TB     = cutlass::float_e4m3_t;
using AccumT = float;                    // FP32 累加/输出

// ============================================================================
// 测试 Config 列表
// ============================================================================
// v1: 128×128 tile, kTileK=128, stage=3
using FP8Config_v1 = gemm_fp8_sm90::GemmConfigFP8<TA, TB, 128, 128, 128, 3>;
// v2: 128×256 tile, kTileK=128, stage=3（N tile 更大，但 255 regs + spill）
using FP8Config_v2 = gemm_fp8_sm90::GemmConfigFP8<TA, TB, 128, 256, 128, 3>;
// v3: 128×128 tile, kTileK=128, stage=4（SMEM 128KB，验证多 stage 效果）
using FP8Config_v3 = gemm_fp8_sm90::GemmConfigFP8<TA, TB, 128, 128, 128, 4>;
// v4: 与 v1 相同，使用 block swizzle 版本 kernel（改善 L2 命中率）
using FP8Config_v4 = gemm_fp8_sm90::GemmConfigFP8<TA, TB, 128, 128, 128, 3>;
// v5: Ping-Pong kernel（Consumer WG 128线程 + Producer 1 warp 32线程 = 160线程）
using FP8Config_v5 = gemm_fp8_sm90::GemmConfigFP8<TA, TB, 128, 128, 128, 3>;
// v7: Persistent Ping-Pong（消除 wave quantization，grid=num_SMs×2）
using FP8Config_v7 = gemm_fp8_sm90::GemmConfigFP8<TA, TB, 128, 128, 128, 3>;

// ============================================================================
// 错误检查宏
// ============================================================================
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_err));          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CUBLASLT_CHECK(call)                                                \
    do {                                                                    \
        cublasStatus_t _s = (call);                                         \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLASLt error at %s:%d: status=%d\n",        \
                    __FILE__, __LINE__, (int)_s);                           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================================
// 数据生成
// ============================================================================
void gen_rand_fp8(TA* data, size_t n, float range = 0.5f) {
    std::default_random_engine rng(42);
    std::uniform_real_distribution<float> dist(-range, range);
    for (size_t i = 0; i < n; ++i)
        data[i] = static_cast<TA>(dist(rng));
}

// ============================================================================
// FP32 cuBLAS Ground Truth
// 将 FP8 先转 FP32，然后用 cublasSgemm 计算 C = A * B^T
// ============================================================================
void run_cublas_fp32_ref(
    TA* h_A, TB* h_B, AccumT* h_C_ref,
    int M, int N, int K)
{
    std::vector<float> h_A_f32(M * K), h_B_f32(N * K);
    for (int i = 0; i < M * K; ++i) h_A_f32[i] = static_cast<float>(h_A[i]);
    for (int i = 0; i < N * K; ++i) h_B_f32[i] = static_cast<float>(h_B[i]);

    float *d_A_f32, *d_B_f32, *d_C_f32;
    CUDA_CHECK(cudaMalloc(&d_A_f32, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_f32, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_f32, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A_f32, h_A_f32.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_f32, h_B_f32.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    // Row-major C = A * B^T → cuBLAS col-major: C^T = B * A^T
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B_f32, K,
                d_A_f32, K,
                &beta,
                d_C_f32, N);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_ref, d_C_f32, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A_f32));
    CUDA_CHECK(cudaFree(d_B_f32));
    CUDA_CHECK(cudaFree(d_C_f32));
}

// ============================================================================
// cuBLASLt FP8 GEMM — 预创建描述符，benchmark 只调 cublasLtMatmul
// ============================================================================
struct CublasLtFP8Context {
    cublasLtHandle_t           ltHandle;
    cublasLtMatmulDesc_t       operationDesc;
    cublasLtMatrixLayout_t     Adesc, Bdesc, Cdesc;
    cublasLtMatmulAlgo_t       algo;
    void*                      d_workspace;
    static constexpr size_t    kWorkspaceSize = 32ULL * 1024 * 1024;
    float                      alpha = 1.0f;
    float                      beta  = 0.0f;

    void init(cublasLtHandle_t handle, int M, int N, int K) {
        ltHandle = handle;

        cublasOperation_t transa = CUBLAS_OP_N;
        cublasOperation_t transb = CUBLAS_OP_T;

        CUBLASLT_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc,
            CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc,
            CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

        cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, M, K, K));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, N, K, K));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, N));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

        CUDA_CHECK(cudaMalloc(&d_workspace, kWorkspaceSize));

        cublasLtMatmulPreference_t preference = nullptr;
        CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceSize, sizeof(kWorkspaceSize)));

        cublasLtMatmulHeuristicResult_t heuristicResult = {};
        int returnedResults = 0;
        CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
            ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc,
            preference, 1, &heuristicResult, &returnedResults));

        if (returnedResults == 0) {
            fprintf(stderr, "[ERROR] cuBLASLt FP8: No valid algorithm found!\n");
            exit(EXIT_FAILURE);
        }
        algo = heuristicResult.algo;
        cublasLtMatmulPreferenceDestroy(preference);
    }

    void run(TA* d_A, TB* d_B, AccumT* d_C) {
        CUBLASLT_CHECK(cublasLtMatmul(
            ltHandle, operationDesc,
            &alpha, d_A, Adesc,
                    d_B, Bdesc,
            &beta,  d_C, Cdesc,
                    d_C, Cdesc,
            &algo,
            d_workspace, kWorkspaceSize, 0));
    }

    void destroy() {
        cublasLtMatrixLayoutDestroy(Cdesc);
        cublasLtMatrixLayoutDestroy(Bdesc);
        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatmulDescDestroy(operationDesc);
        cudaFree(d_workspace);
    }
};

// ============================================================================
// 正确性检验
// ============================================================================
bool check_result(const AccumT* h_ref, AccumT* d_res, int N,
                  const char* label, float rel_threshold = 0.03f)
{
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<AccumT> h_res(N);
    CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, N * sizeof(AccumT), cudaMemcpyDeviceToHost));

    double max_diff = 0.0, max_ref = 0.0;
    int err_cnt = 0;
    for (int i = 0; i < N; ++i) {
        double vr   = std::abs((double)h_ref[i]);
        double diff = std::abs((double)h_ref[i] - (double)h_res[i]);
        if (vr   > max_ref)  max_ref  = vr;
        if (diff > max_diff) max_diff = diff;
        if (diff > rel_threshold * vr + 1e-3 && err_cnt < 3) {
            printf("  [%s] Error@%d: ref=%.5f res=%.5f\n", label, i, h_ref[i], h_res[i]);
            ++err_cnt;
        }
    }
    double rel = (max_ref > 1e-6) ? max_diff / max_ref : max_diff;
    bool ok = (rel < rel_threshold);
    printf("[%s] %s  max_diff=%.4e  max_ref=%.4e  rel_err=%.2e  (threshold=%.2e)\n",
           label, ok ? "PASSED ✓" : "FAILED ✗",
           max_diff, max_ref, rel, (double)rel_threshold);
    return ok;
}

// ============================================================================
// 性能 Benchmark
// ============================================================================
template <typename KernelFunc>
float benchmark(KernelFunc func, int n_warmup = 5, int n_iter = 20) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < n_warmup; ++i) func();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < n_iter; ++i) func();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / n_iter;
}

double tflops(int M, int N, int K, float ms) {
    return (2.0 * M * N * K) * 1e-12 / (ms * 1e-3);
}

void print_sep(const char* s) {
    printf("\n=================================================================\n");
    printf("  %s\n", s);
    printf("=================================================================\n");
}

// ============================================================================
// 通用 FP8 TMA Kernel 测试辅助（标准版：自然 block 顺序）
// ============================================================================
template <typename Config>
float run_fp8_kernel(
    TA* d_A, TB* d_B,
    AccumT* d_C_out,
    const AccumT* h_C_cublas,
    const AccumT* h_C_ref,
    int M, int N, int K,
    size_t bytes_C,
    const char* label)
{
    typename Config::SmemLayoutAtomA smem_atom_a;
    typename Config::SmemLayoutAtomB smem_atom_b;

    auto gmem_A = make_tensor(make_gmem_ptr(d_A),
                              make_layout(make_shape(M, K), make_stride(K, _1{})));
    auto gmem_B = make_tensor(make_gmem_ptr(d_B),
                              make_layout(make_shape(N, K), make_stride(K, _1{})));

    auto tma_a = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, gmem_A, smem_atom_a, cute::Int<1>{});
    auto tma_b = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, gmem_B, smem_atom_b, cute::Int<1>{});

    constexpr size_t smem_size   = gemm_fp8_sm90::get_smem_size_fp8_tma<Config>();
    constexpr size_t smem_size_C =
        cute::cosize(typename Config::SmemLayoutC{}) * sizeof(typename Config::AccumType);
    constexpr size_t smem_size_total = smem_size > smem_size_C ? smem_size : smem_size_C;

    printf("  TileM=%d TileN=%d TileK=%d Stage=%d  SMEM=%.1fKB\n",
           Config::kTileM, Config::kTileN, Config::kTileK, Config::kStage,
           smem_size_total / 1024.0f);

    auto kernel_ptr = gemm_fp8_sm90::gemm_kernel_fp8_tma<Config, decltype(tma_a), decltype(tma_b)>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_total));

    dim3 block(Config::kNumThreads);
    dim3 grid(N / Config::kTileN, M / Config::kTileM);

    auto launch = [&]() {
        kernel_ptr<<<grid, block, smem_size_total>>>(d_C_out, tma_a, tma_b, M, N, K);
    };

    // 正确性
    CUDA_CHECK(cudaMemset(d_C_out, 0, bytes_C));
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_ptr));
        printf("  Kernel registers: %d\n", attr.numRegs);
    }

    char label_ref[128], label_lt[128];
    snprintf(label_ref, sizeof(label_ref), "%s vs FP32-ref", label);
    snprintf(label_lt,  sizeof(label_lt),  "%s vs cuBLASLt", label);
    check_result(h_C_ref,    d_C_out, M * N, label_ref, 0.03f);
    check_result(h_C_cublas, d_C_out, M * N, label_lt,  0.005f);

    float ms = benchmark(launch);
    printf("  %s: %.3f ms   %.2f TFLOPS\n", label, ms, tflops(M, N, K, ms));
    return ms;
}

// ============================================================================
// FP8 TMA Kernel v4 (Block Swizzle)
// ============================================================================
template <typename Config, int kGroupSize = 8>
float run_fp8_kernel_swizzle(
    TA* d_A, TB* d_B,
    AccumT* d_C_out,
    const AccumT* h_C_cublas,
    const AccumT* h_C_ref,
    int M, int N, int K,
    size_t bytes_C,
    const char* label)
{
    typename Config::SmemLayoutAtomA smem_atom_a;
    typename Config::SmemLayoutAtomB smem_atom_b;

    auto gmem_A = make_tensor(make_gmem_ptr(d_A),
                              make_layout(make_shape(M, K), make_stride(K, _1{})));
    auto gmem_B = make_tensor(make_gmem_ptr(d_B),
                              make_layout(make_shape(N, K), make_stride(K, _1{})));

    auto tma_a = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, gmem_A, smem_atom_a, cute::Int<1>{});
    auto tma_b = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, gmem_B, smem_atom_b, cute::Int<1>{});

    constexpr size_t smem_size   = gemm_fp8_sm90::get_smem_size_fp8_tma<Config>();
    constexpr size_t smem_size_C =
        cute::cosize(typename Config::SmemLayoutC{}) * sizeof(typename Config::AccumType);
    constexpr size_t smem_size_total = smem_size > smem_size_C ? smem_size : smem_size_C;

    printf("  TileM=%d TileN=%d TileK=%d Stage=%d  SMEM=%.1fKB  GroupSize=%d\n",
           Config::kTileM, Config::kTileN, Config::kTileK, Config::kStage,
           smem_size_total / 1024.0f, kGroupSize);

    using KernelType = decltype(gemm_fp8_sm90::gemm_kernel_fp8_tma_swizzle<
        Config, decltype(tma_a), decltype(tma_b), kGroupSize>);
    auto kernel_ptr = gemm_fp8_sm90::gemm_kernel_fp8_tma_swizzle<
        Config, decltype(tma_a), decltype(tma_b), kGroupSize>;

    CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_total));

    dim3 block(Config::kNumThreads);
    dim3 grid(N / Config::kTileN, M / Config::kTileM);

    auto launch = [&]() {
        kernel_ptr<<<grid, block, smem_size_total>>>(d_C_out, tma_a, tma_b, M, N, K);
    };

    // 正确性
    CUDA_CHECK(cudaMemset(d_C_out, 0, bytes_C));
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_ptr));
        printf("  Kernel registers: %d\n", attr.numRegs);
    }

    char label_ref[128], label_lt[128];
    snprintf(label_ref, sizeof(label_ref), "%s vs FP32-ref", label);
    snprintf(label_lt,  sizeof(label_lt),  "%s vs cuBLASLt", label);
    check_result(h_C_ref,    d_C_out, M * N, label_ref, 0.03f);
    check_result(h_C_cublas, d_C_out, M * N, label_lt,  0.005f);

    float ms = benchmark(launch);
    printf("  %s: %.3f ms   %.2f TFLOPS\n", label, ms, tflops(M, N, K, ms));
    return ms;
}

// ============================================================================
// FP8 Ping-Pong Kernel v5 (Consumer __forceinline__ + Producer __noinline__)
// ============================================================================
template <typename Config>
float run_fp8_kernel_pingpong(
    TA* d_A, TB* d_B,
    AccumT* d_C_out,
    const AccumT* h_C_cublas,
    const AccumT* h_C_ref,
    int M, int N, int K,
    size_t bytes_C,
    const char* label)
{
    typename Config::SmemLayoutAtomA smem_atom_a;
    typename Config::SmemLayoutAtomB smem_atom_b;

    auto gmem_A = make_tensor(make_gmem_ptr(d_A),
                              make_layout(make_shape(M, K), make_stride(K, _1{})));
    auto gmem_B = make_tensor(make_gmem_ptr(d_B),
                              make_layout(make_shape(N, K), make_stride(K, _1{})));

    auto tma_a = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, gmem_A, smem_atom_a, cute::Int<1>{});
    auto tma_b = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, gmem_B, smem_atom_b, cute::Int<1>{});

    // Ping-pong SMEM: AB buf + 2×kStage mbarriers (mbar_full + mbar_empty)
    constexpr size_t smem_AB   = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(typename Config::TA)
                                + cute::cosize(typename Config::SmemLayoutB{}) * sizeof(typename Config::TB);
    constexpr size_t mbar_off  = (smem_AB + 7) & ~7;
    constexpr size_t smem_size = mbar_off + 2 * Config::kStage * sizeof(uint64_t);

    printf("  [PP] TileM=%d TileN=%d TileK=%d Stage=%d  SMEM=%.1fKB  Threads=%d\n",
           Config::kTileM, Config::kTileN, Config::kTileK, Config::kStage,
           smem_size / 1024.0f, Config::kNumThreadsPP);

    auto kernel_ptr = gemm_fp8_sm90::gemm_kernel_fp8_pingpong<Config, decltype(tma_a), decltype(tma_b)>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 block(Config::kNumThreadsPP);
    dim3 grid(N / Config::kTileN, M / Config::kTileM);

    auto launch = [&]() {
        kernel_ptr<<<grid, block, smem_size>>>(d_C_out, tma_a, tma_b, M, N, K);
    };

    // 正确性
    CUDA_CHECK(cudaMemset(d_C_out, 0, bytes_C));
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_ptr));
        printf("  [PP] Kernel registers: %d\n", attr.numRegs);
    }

    char label_ref[128], label_lt[128];
    snprintf(label_ref, sizeof(label_ref), "%s vs FP32-ref", label);
    snprintf(label_lt,  sizeof(label_lt),  "%s vs cuBLASLt", label);
    check_result(h_C_ref,    d_C_out, M * N, label_ref, 0.03f);
    check_result(h_C_cublas, d_C_out, M * N, label_lt,  0.005f);

    float ms = benchmark(launch);
    printf("  %s: %.3f ms   %.2f TFLOPS\n", label, ms, tflops(M, N, K, ms));
    return ms;
}

// ============================================================================
// FP8 Persistent Ping-Pong Kernel v7
// grid = num_SMs × occupancy_per_SM（持久化调度，消除 wave quantization）
// ============================================================================
template <typename Config>
float run_fp8_kernel_persistent(
    TA* d_A, TB* d_B,
    AccumT* d_C_out,
    const AccumT* h_C_cublas,
    const AccumT* h_C_ref,
    int M, int N, int K,
    size_t bytes_C,
    const char* label)
{
    typename Config::SmemLayoutAtomA smem_atom_a;
    typename Config::SmemLayoutAtomB smem_atom_b;

    auto gmem_A = make_tensor(make_gmem_ptr(d_A),
                              make_layout(make_shape(M, K), make_stride(K, _1{})));
    auto gmem_B = make_tensor(make_gmem_ptr(d_B),
                              make_layout(make_shape(N, K), make_stride(K, _1{})));

    auto tma_a = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, gmem_A, smem_atom_a, cute::Int<1>{});
    auto tma_b = cute::make_tma_copy(cute::SM90_TMA_LOAD{}, gmem_B, smem_atom_b, cute::Int<1>{});

    // SMEM: AB + 2×kStage mbarriers (full + empty) + 4 bytes tile_id
    constexpr size_t smem_AB   = cute::cosize(typename Config::SmemLayoutA{}) * sizeof(typename Config::TA)
                                + cute::cosize(typename Config::SmemLayoutB{}) * sizeof(typename Config::TB);
    constexpr size_t mbar_off  = (smem_AB + 7) & ~7;
    constexpr size_t smem_size = mbar_off + 2 * Config::kStage * sizeof(uint64_t)
                                + sizeof(int);  // tile_id_ptr 存储在 mbar 之后

    // 查询 SM 数量，计算 persistent grid size
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int num_sms = prop.multiProcessorCount;
    int occupancy_per_sm = 2;  // 目标 occupancy=2
    int persistent_grid = num_sms * occupancy_per_sm;

    int num_m_tiles = M / Config::kTileM;
    int num_n_tiles = N / Config::kTileN;
    int total_tiles = num_m_tiles * num_n_tiles;

    printf("  [PP-Persistent] TileM=%d TileN=%d TileK=%d Stage=%d  SMEM=%.1fKB\n",
           Config::kTileM, Config::kTileN, Config::kTileK, Config::kStage,
           smem_size / 1024.0f);
    printf("  [PP-Persistent] SMs=%d, persistent_grid=%d, total_tiles=%d\n",
           num_sms, persistent_grid, total_tiles);

    auto kernel_ptr = gemm_fp8_sm90::gemm_kernel_fp8_persistent<
        Config, decltype(tma_a), decltype(tma_b)>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // Tile counter (全局 atomic，初始化为 0)
    int* d_tile_counter;
    CUDA_CHECK(cudaMalloc(&d_tile_counter, sizeof(int)));

    dim3 block(Config::kNumThreadsPP);
    dim3 grid_dim(persistent_grid);

    auto launch = [&]() {
        CUDA_CHECK(cudaMemset(d_tile_counter, 0, sizeof(int)));
        kernel_ptr<<<grid_dim, block, smem_size>>>(
            d_C_out, tma_a, tma_b,
            d_tile_counter, total_tiles, num_n_tiles,
            M, N, K);
    };

    // 正确性
    CUDA_CHECK(cudaMemset(d_C_out, 0, bytes_C));
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_ptr));
        printf("  [PP-Persistent] Kernel registers: %d\n", attr.numRegs);
    }

    char label_ref[128], label_lt[128];
    snprintf(label_ref, sizeof(label_ref), "%s vs FP32-ref", label);
    snprintf(label_lt,  sizeof(label_lt),  "%s vs cuBLASLt", label);
    check_result(h_C_ref,    d_C_out, M * N, label_ref, 0.03f);
    check_result(h_C_cublas, d_C_out, M * N, label_lt,  0.005f);

    float ms = benchmark(launch);
    printf("  %s: %.3f ms   %.2f TFLOPS\n", label, ms, tflops(M, N, K, ms));

    CUDA_CHECK(cudaFree(d_tile_counter));
    return ms;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {

    // ---- 检查架构 ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  SM %d.%d\n", prop.name, prop.major, prop.minor);
    if (prop.major < 9) {
        printf("ERROR: SM90 (Hopper) required for FP8 wgmma. Current: SM%d%d\n",
               prop.major, prop.minor);
        return -1;
    }

    // ---- 问题规模 ----
    int M = (argc > 1) ? std::atoi(argv[1]) : 4096;
    int N = (argc > 2) ? std::atoi(argv[2]) : 4096;
    int K = (argc > 3) ? std::atoi(argv[3]) : 2048;

    // 对齐要求: M 128的倍数, N 256的倍数（v2用到），K 128的倍数
    if (M % 128 != 0 || N % 256 != 0 || K % 128 != 0) {
        printf("ERROR: M must be multiple of 128, N multiple of 256, K multiple of 128\n");
        printf("       Provided: M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }

    printf("FP8 GEMM (E4M3×E4M3, FP32 out): M=%d N=%d K=%d\n\n", M, N, K);

    // ---- 内存分配 ----
    size_t bytes_A = (size_t)M * K * sizeof(TA);
    size_t bytes_B = (size_t)N * K * sizeof(TB);
    size_t bytes_C = (size_t)M * N * sizeof(AccumT);

    std::vector<TA>     h_A(M * K);
    std::vector<TB>     h_B(N * K);
    std::vector<AccumT> h_C_ref(M * N);

    gen_rand_fp8(h_A.data(), M * K);
    gen_rand_fp8(h_B.data(), N * K);

    TA     *d_A, *d_B;
    AccumT *d_C_cublas, *d_C_v1, *d_C_v2, *d_C_v3, *d_C_v4, *d_C_v5, *d_C_v7;

    CUDA_CHECK(cudaMalloc(&d_A,         bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B,         bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C_cublas,  bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_v1,      bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_v2,      bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_v3,      bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_v4,      bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_v5,      bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_v7,      bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    // =========================================================================
    // 0. FP32 cuBLAS Ground Truth
    // =========================================================================
    print_sep("0. FP32 cuBLAS Ground Truth (FP8 → FP32 → cuBLAS)");
    run_cublas_fp32_ref(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    printf("Ground truth computed.\n");

    // =========================================================================
    // 1. cuBLASLt FP8 Baseline
    //    预创建描述符/算法/workspace，benchmark 只调 cublasLtMatmul
    // =========================================================================
    print_sep("1. cuBLASLt FP8 Baseline");
    cublasLtHandle_t ltHandle;
    CUBLASLT_CHECK(cublasLtCreate(&ltHandle));

    CublasLtFP8Context lt_ctx;
    lt_ctx.init(ltHandle, M, N, K);

    // 正确性
    lt_ctx.run(d_A, d_B, d_C_cublas);
    CUDA_CHECK(cudaDeviceSynchronize());
    check_result(h_C_ref.data(), d_C_cublas, M * N, "cuBLASLt FP8 vs FP32-ref", 0.01f);

    // 性能（预热后再计时）
    float ms_cublas = benchmark([&](){ lt_ctx.run(d_A, d_B, d_C_cublas); });
    printf("cuBLASLt FP8:  %.3f ms   %.2f TFLOPS\n", ms_cublas, tflops(M, N, K, ms_cublas));

    // 保存 cuBLASLt 输出到 host（供后续 kernel 对比）
    std::vector<AccumT> h_C_cublas(M * N);
    lt_ctx.run(d_A, d_B, d_C_cublas);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_cublas.data(), d_C_cublas, bytes_C, cudaMemcpyDeviceToHost));

    // =========================================================================
    // 2. FP8 TMA v1: 128×128×128, stage=3（自然 block 顺序）
    // =========================================================================
    print_sep("2. FP8 TMA v1: TileM=128 TileN=128 TileK=128 Stage=3 (single WG)");
    float ms_v1 = run_fp8_kernel<FP8Config_v1>(
        d_A, d_B, d_C_v1, h_C_cublas.data(), h_C_ref.data(),
        M, N, K, bytes_C, "FP8 TMA v1");

    // =========================================================================
    // 3. FP8 TMA v2: 128×256×128, stage=3（N tile 更大）
    // =========================================================================
    print_sep("3. FP8 TMA v2: TileM=128 TileN=256 TileK=128 Stage=3 (register spill test)");
    float ms_v2 = run_fp8_kernel<FP8Config_v2>(
        d_A, d_B, d_C_v2, h_C_cublas.data(), h_C_ref.data(),
        M, N, K, bytes_C, "FP8 TMA v2");

    // =========================================================================
    // 4. FP8 TMA v3: 128×128×128, stage=4
    // =========================================================================
    print_sep("4. FP8 TMA v3: TileM=128 TileN=128 TileK=128 Stage=4 (single WG)");
    float ms_v3 = run_fp8_kernel<FP8Config_v3>(
        d_A, d_B, d_C_v3, h_C_cublas.data(), h_C_ref.data(),
        M, N, K, bytes_C, "FP8 TMA v3");

    // =========================================================================
    // 5. FP8 TMA v4: 128×128×128, stage=3 + Block Swizzle
    // =========================================================================
    print_sep("5. FP8 TMA v4: TileM=128 TileN=128 TileK=128 Stage=3 + Block Swizzle");
    float ms_v4 = run_fp8_kernel_swizzle<FP8Config_v4, 8>(
        d_A, d_B, d_C_v4, h_C_cublas.data(), h_C_ref.data(),
        M, N, K, bytes_C, "FP8 TMA v4");

    // =========================================================================
    // 6. FP8 Ping-Pong v5: 128×128×128, stage=3 (Consumer WG + Producer Warp)
    // =========================================================================
    print_sep("6. FP8 Ping-Pong v5: TileM=128 TileN=128 TileK=128 Stage=3 (PP)");
    float ms_v5 = run_fp8_kernel_pingpong<FP8Config_v5>(
        d_A, d_B, d_C_v5, h_C_cublas.data(), h_C_ref.data(),
        M, N, K, bytes_C, "FP8 PP v5");

    // =========================================================================
    // 7. FP8 Persistent PP v7: Ping-Pong + Persistent Scheduling
    //    grid = num_SMs × 2，消除 wave quantization 损失
    // =========================================================================
    print_sep("7. FP8 Persistent PP v7: TileM=128 TileN=128 TileK=128 Stage=3 (Persistent)");
    float ms_v7 = run_fp8_kernel_persistent<FP8Config_v7>(
        d_A, d_B, d_C_v7, h_C_cublas.data(), h_C_ref.data(),
        M, N, K, bytes_C, "FP8 PP-Persistent v7");

    // =========================================================================
    // 汇总
    // =========================================================================
    print_sep("Summary");
    printf("%-40s %8s   %8s   %s\n", "Kernel", "ms", "TFLOPS", "vs cuBLASLt");
    printf("%-40s %8.3f   %8.2f   ---\n",
           "cuBLASLt FP8", ms_cublas, tflops(M, N, K, ms_cublas));
    printf("%-40s %8.3f   %8.2f   %+.1f%%\n",
           "v1 (128×128×128 s3)",
           ms_v1, tflops(M, N, K, ms_v1), 100.0 * (ms_cublas / ms_v1 - 1.0));
    printf("%-40s %8.3f   %8.2f   %+.1f%%  [255 regs, 3KB spill]\n",
           "v2 (128×256×128 s3)",
           ms_v2, tflops(M, N, K, ms_v2), 100.0 * (ms_cublas / ms_v2 - 1.0));
    printf("%-40s %8.3f   %8.2f   %+.1f%%\n",
           "v3 (128×128×128 s4)",
           ms_v3, tflops(M, N, K, ms_v3), 100.0 * (ms_cublas / ms_v3 - 1.0));
    printf("%-40s %8.3f   %8.2f   %+.1f%%  [block swizzle g=%d]\n",
           "v4 (128×128×128 s3+swizzle)",
           ms_v4, tflops(M, N, K, ms_v4), 100.0 * (ms_cublas / ms_v4 - 1.0), 8);
    printf("%-40s %8.3f   %8.2f   %+.1f%%  [ping-pong, 160 threads]\n",
           "v5 (128×128×128 s3+ping-pong)",
           ms_v5, tflops(M, N, K, ms_v5), 100.0 * (ms_cublas / ms_v5 - 1.0));
    printf("%-40s %8.3f   %8.2f   %+.1f%%  [persistent, grid=%d]\n",
           "v7 (128×128×128 s3+persistent-PP)",
           ms_v7, tflops(M, N, K, ms_v7), 100.0 * (ms_cublas / ms_v7 - 1.0),
           0);  // grid size printed in kernel output

    // ---- 清理 ----
    lt_ctx.destroy();
    cublasLtDestroy(ltHandle);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_cublas));
    CUDA_CHECK(cudaFree(d_C_v1));
    CUDA_CHECK(cudaFree(d_C_v2));
    CUDA_CHECK(cudaFree(d_C_v3));
    CUDA_CHECK(cudaFree(d_C_v4));
    CUDA_CHECK(cudaFree(d_C_v5));
    CUDA_CHECK(cudaFree(d_C_v7));

    return 0;
}
