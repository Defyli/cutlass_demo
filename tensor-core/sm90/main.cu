#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>

// CuTe headers
#include "cute/tensor.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"

#include "gemm_sm90.cuh"

// ============================================================================
// 类型定义
// ============================================================================
using T      = cute::bfloat16_t;  // BF16 输入
using AccumT = float;              // FP32 累加/输出

// ============================================================================
// 测试 Config: 128x128 tile, K=64, 3 stages, 128 threads
// ============================================================================
using TestConfig = gemm_sm90::GemmConfig<T, 128, 128, 64, 3>;

// 单WG TMA kStage=4: 增大预取窗口，与 kStage=3 对比是否有收益
//   SMEM: (128*64*4 + 128*64*4)*2B + 4*8B = 131072 + 32 = 128KB
using TestConfig4 = gemm_sm90::GemmConfig<T, 128, 128, 64, 4>;

// Ping-Pong 专用 Config: kStage=4
//   Producer 可以比 Consumer 多预取 kStage-1=3 个 tile, 充分隐藏 TMA 延迟
//   SMEM: (128*64*4 + 128*64*4) * 2B + 2*4*8B = 131072 + 64 = 128KB (< 227KB)
using PingPongConfig = gemm_sm90::GemmConfig<T, 128, 128, 64, 4>;

// Ping-Pong kStage=3 对照组: 与单WG TMA kStage=3 同等 SMEM(96KB)
//   用于隔离 "SMEM 增大导致 occupancy 下降" vs "Ping-Pong 同步开销" 两个变量
//   SMEM: (128*64*3 + 128*64*3) * 2B + 2*3*8B = 98304 + 48 = 96KB
using PingPongConfig3 = gemm_sm90::GemmConfig<T, 128, 128, 64, 3>;

// Ping-Pong kStage=2 极限测试: 最小 SMEM(64KB), 理论 occupancy=3
//   SMEM: (128*64*2 + 128*64*2) * 2B + 2*2*8B = 65536 + 32 = 64KB
//   Occupancy: floor(227/64) = 3 blocks/SM ← 比 kStage=3 的 2 blocks 更高!
//   目的: 验证是否更高 occupancy 能带来更好性能
using PingPongConfig2 = gemm_sm90::GemmConfig<T, 128, 128, 64, 2>;

// ─────────────────────────────────────────────────────────────────────────────
// 新增 kernel 的 Config (基于 hpc-ops 学习)
// ─────────────────────────────────────────────────────────────────────────────

// 版本 A (v6): Ping-Pong + TMA Store Epilogue (160 线程, FP32 output)
//   kStage=2: A(32) + B(32) + mbar + C_float(64) ≈ 128KB ✓
//   (kStage=3 时: A(48)+B(48)+C_float(64) = 160KB > 128KB, 超限)
using PPTmaStoreConfig = gemm_sm90::GemmConfig<T, 128, 128, 64, 2>;

// 版本 B (2WG): 384 线程 双 WG Cooperative + FP32 直接写回
//   kStage=2: A(32) + B0(32) + B1(32) + mbar ≈ 96KB ✓
//   (FP32 输出直接 STG 写 GMEM, 无 C buffer)
using Config2WG = gemm_sm90::GemmConfig<T, 128, 128, 64, 2>;

// ============================================================================
// CUDA / cuBLAS 错误检查宏
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

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t _s = (call);                                         \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n",          \
                    __FILE__, __LINE__, (int)_s);                           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================================
// 辅助函数
// ============================================================================
void gen_rand_bf16(T* data, size_t n, float range = 0.01f) {
    std::default_random_engine rng(42);
    std::uniform_real_distribution<float> dist(-range, range);
    for (size_t i = 0; i < n; ++i)
        data[i] = static_cast<T>(dist(rng));
}

// cuBLAS BF16 GEMM: C(float) = A(bf16, MxK row) * B(bf16, NxK row)^T
void run_cublas_bf16(cublasHandle_t handle,
                     T* d_A, T* d_B, AccumT* d_C,
                     int M, int N, int K)
{
    float alpha = 1.0f, beta = 0.0f;
    // cuBLAS col-major: C^T = B * A^T  (col-major NxM)
    // B (NxK row) -> 视为 KxN col -> CUBLAS_OP_T: NxK
    // A (MxK row) -> 视为 KxM col -> CUBLAS_OP_N: KxM = A^T
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_16BF, K,
        d_A, CUDA_R_16BF, K,
        &beta,
        d_C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// 验证结果 (比对 device 结果与 host reference, FP32)
bool check_result(const AccumT* h_ref, AccumT* d_res, int N,
                  const char* label, float rel_threshold = 0.05f)
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
        if (diff > rel_threshold && err_cnt < 3) {
            printf("  [%s] Error@%d: ref=%.5f res=%.5f\n", label, i, h_ref[i], h_res[i]);
            ++err_cnt;
        }
    }
    double rel = (max_ref > 1e-6) ? max_diff / max_ref : max_diff;
    bool ok = (rel < rel_threshold);
    printf("[%s] %s  max_diff=%.4e  max_ref=%.4e  rel_err=%.2e\n",
           label, ok ? "PASSED" : "FAILED", max_diff, max_ref, rel);
    return ok;
}

// 性能 Benchmark
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
    printf("\n===============================================================\n");
    printf("  %s\n", s);
    printf("===============================================================\n");
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
        printf("ERROR: SM90 (Hopper) required. Current: SM%d%d\n",
               prop.major, prop.minor);
        return -1;
    }

    // ---- 问题规模 ----
    int M = (argc > 1) ? std::atoi(argv[1]) : 4096;
    int N = (argc > 2) ? std::atoi(argv[2]) : 4096;
    int K = (argc > 3) ? std::atoi(argv[3]) : 2048;

    // 对齐检查
    if (M % 128 != 0 || N % 128 != 0 || K % 64 != 0) {
        printf("ERROR: M/N must be multiple of 128, K must be multiple of 64\n");
        printf("       Provided: M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }

    printf("GEMM: M=%d N=%d K=%d  (BF16 in, FP32 out)\n\n", M, N, K);

    // ---- 内存分配 ----
    size_t bytes_A = (size_t)M * K * sizeof(T);
    size_t bytes_B = (size_t)N * K * sizeof(T);
    size_t bytes_C = (size_t)M * N * sizeof(AccumT);

    std::vector<T>      h_A(M * K), h_B(N * K);
    std::vector<AccumT> h_C_ref(M * N);

    gen_rand_bf16(h_A.data(), M * K);
    gen_rand_bf16(h_B.data(), N * K);

    T      *d_A, *d_B;
    AccumT *d_C_cublas, *d_C_tma, *d_C_cpasync, *d_C_pingpong, *d_C_pingpong3;
    // Section 4 (v6 TMA Store) 和 Section 5 (2WG) 的 FP32 输出 buffer
    AccumT *d_C_v6, *d_C_2wg;

    CUDA_CHECK(cudaMalloc(&d_A,           bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B,           bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C_cublas,    bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_tma,       bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_cpasync,   bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_pingpong,  bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_pingpong3, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_v6,        bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_2wg,       bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // =========================================================================
    // 0. cuBLAS Baseline (用于正确性参考)
    // =========================================================================
    print_sep("0. cuBLAS BF16 Baseline");

    run_cublas_bf16(handle, d_A, d_B, d_C_cublas, M, N, K);
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_cublas, bytes_C, cudaMemcpyDeviceToHost));

    float ms_cublas = benchmark([&](){
        run_cublas_bf16(handle, d_A, d_B, d_C_cublas, M, N, K);
    });
    printf("cuBLAS BF16:    %.3f ms   %.2f TFLOPS\n", ms_cublas, tflops(M, N, K, ms_cublas));

    // =========================================================================
    // 1. CuTe SM90 TMA + GMMA Kernel
    // =========================================================================
    print_sep("1. CuTe SM90: TMA + GMMA");

    {
        using Cfg = TestConfig;

        // ---- 构建 TMA copy 对象 (HOST 端) ----
        // 全局矩阵 tensor (真实指针)
        auto tensor_A = cute::make_tensor(
            cute::make_gmem_ptr(d_A),
            cute::make_shape(M, K),
            cute::make_stride(K, cute::_1{})
        );
        auto tensor_B = cute::make_tensor(
            cute::make_gmem_ptr(d_B),
            cute::make_shape(N, K),
            cute::make_stride(K, cute::_1{})
        );

        // SMEM layout (单个 stage, 即 TMA 每次加载的形状)
        // SmemLayoutAtom = Layout_K_SW128_Atom<T>, 对应 (8, 64) 的原子
        // tile_to_shape 扩展到 (kTileM=128, kTileK=64)
        auto smem_layout_a = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomA{},
            cute::make_shape(cute::Int<Cfg::kTileM>{}, cute::Int<Cfg::kTileK>{})
        );
        auto smem_layout_b = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomB{},
            cute::make_shape(cute::Int<Cfg::kTileN>{}, cute::Int<Cfg::kTileK>{})
        );

        // make_tma_copy: 给定 gmem tensor 和 smem layout, 构建 TMA copy 对象
        //   - 内部构建 CUtensorMap descriptor
        //   - cluster_size=1: 不使用 multicast
        auto tma_a = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{},
            tensor_A,
            smem_layout_a,
            cute::Int<1>{}  // cluster_size
        );
        auto tma_b = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{},
            tensor_B,
            smem_layout_b,
            cute::Int<1>{}
        );

        // ---- SMEM 大小 ----
        constexpr size_t smem_total = gemm_sm90::get_smem_size_tma<Cfg>();
        printf("SMEM: A=%zuB B=%zuB mbar=%zuB total=%zuB (%.1fKB)\n",
               cute::cosize(typename Cfg::SmemLayoutA{}) * sizeof(T),
               cute::cosize(typename Cfg::SmemLayoutB{}) * sizeof(T),
               (size_t)Cfg::kStage * sizeof(uint64_t),
               smem_total, smem_total / 1024.0);
        printf("Device max dynamic SMEM: %zuKB\n",
               prop.sharedMemPerBlockOptin / 1024);

        if (smem_total > prop.sharedMemPerBlockOptin) {
            printf("SKIP: Required SMEM (%zu) > device max (%zu)\n",
                   smem_total, prop.sharedMemPerBlockOptin);
        } else {
            auto kernel = gemm_sm90::gemm_kernel_tma<Cfg, decltype(tma_a), decltype(tma_b)>;
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_total));

            dim3 block(Cfg::kNumThreads);
            dim3 grid(N / Cfg::kTileN, M / Cfg::kTileM);

            // Correctness check
            CUDA_CHECK(cudaMemset(d_C_tma, 0, bytes_C));
            kernel<<<grid, block, smem_total>>>(d_C_tma, tma_a, tma_b, M, N, K);
            CUDA_CHECK(cudaGetLastError());
            check_result(h_C_ref.data(), d_C_tma, M * N, "TMA+GMMA");

            // Performance
            float ms = benchmark([&](){
                kernel<<<grid, block, smem_total>>>(d_C_tma, tma_a, tma_b, M, N, K);
            });
            printf("SM90 TMA+GMMA:  %.3f ms   %.2f TFLOPS   (vs cuBLAS: %.1f%%)\n",
                   ms, tflops(M, N, K, ms), ms_cublas / ms * 100.f);
        }
    }

    // =========================================================================
    // 1b. CuTe SM90 TMA + GMMA Kernel (kStage=4, 更大预取窗口)
    // =========================================================================
    print_sep("1b. CuTe SM90: TMA + GMMA (kStage=4)");

    {
        using Cfg = TestConfig4;

        auto tensor_A4 = cute::make_tensor(
            cute::make_gmem_ptr(d_A),
            cute::make_shape(M, K),
            cute::make_stride(K, cute::_1{})
        );
        auto tensor_B4 = cute::make_tensor(
            cute::make_gmem_ptr(d_B),
            cute::make_shape(N, K),
            cute::make_stride(K, cute::_1{})
        );
        auto smem_layout_a4 = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomA{},
            cute::make_shape(cute::Int<Cfg::kTileM>{}, cute::Int<Cfg::kTileK>{})
        );
        auto smem_layout_b4 = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomB{},
            cute::make_shape(cute::Int<Cfg::kTileN>{}, cute::Int<Cfg::kTileK>{})
        );
        auto tma_a4 = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{}, tensor_A4, smem_layout_a4, cute::Int<1>{}
        );
        auto tma_b4 = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{}, tensor_B4, smem_layout_b4, cute::Int<1>{}
        );

        constexpr size_t smem4 = gemm_sm90::get_smem_size_tma<Cfg>();
        printf("SMEM: %.1fKB\n", smem4 / 1024.0);

        if (smem4 > prop.sharedMemPerBlockOptin) {
            printf("SKIP: Required SMEM (%zu) > device max (%zu)\n",
                   smem4, prop.sharedMemPerBlockOptin);
        } else {
            auto kernel4 = gemm_sm90::gemm_kernel_tma<Cfg, decltype(tma_a4), decltype(tma_b4)>;
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel4,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem4));

            dim3 block4(Cfg::kNumThreads);
            dim3 grid4(N / Cfg::kTileN, M / Cfg::kTileM);

            CUDA_CHECK(cudaMemset(d_C_tma, 0, bytes_C));
            kernel4<<<grid4, block4, smem4>>>(d_C_tma, tma_a4, tma_b4, M, N, K);
            CUDA_CHECK(cudaGetLastError());
            check_result(h_C_ref.data(), d_C_tma, M * N, "TMA+GMMA kStage=4");

            float ms4 = benchmark([&](){
                kernel4<<<grid4, block4, smem4>>>(d_C_tma, tma_a4, tma_b4, M, N, K);
            });
            printf("SM90 TMA+GMMA(k4): %.3f ms   %.2f TFLOPS   (vs cuBLAS: %.1f%%)\n",
                   ms4, tflops(M, N, K, ms4), ms_cublas / ms4 * 100.f);
        }
    }

    // =========================================================================
    // 2. CuTe SM90 cp.async + GMMA Kernel (无 TMA)
    // =========================================================================
    print_sep("2. CuTe SM90: cp.async + GMMA (无TMA, 对比版)");

    {
        using Cfg = TestConfig;

        constexpr size_t smem_cp = gemm_sm90::get_smem_size_cp_async<Cfg>();
        printf("SMEM: %.1f KB\n", smem_cp / 1024.0);

        auto kernel_cp = gemm_sm90::gemm_kernel_cp_async<Cfg>;
        CUDA_CHECK(cudaFuncSetAttribute(
            kernel_cp,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_cp));

        dim3 block(Cfg::kNumThreads);
        dim3 grid(N / Cfg::kTileN, M / Cfg::kTileM);

        CUDA_CHECK(cudaMemset(d_C_cpasync, 0, bytes_C));
        kernel_cp<<<grid, block, smem_cp>>>(d_C_cpasync, d_A, d_B, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        check_result(h_C_ref.data(), d_C_cpasync, M * N, "cpasync+GMMA");

        float ms = benchmark([&](){
            kernel_cp<<<grid, block, smem_cp>>>(d_C_cpasync, d_A, d_B, M, N, K);
        });
        printf("SM90 cp.async+GMMA: %.3f ms   %.2f TFLOPS   (vs cuBLAS: %.1f%%)\n",
               ms, tflops(M, N, K, ms), ms_cublas / ms * 100.f);
    }

    // =========================================================================
    // 3. CuTe SM90 Ping-Pong TMA + GMMA Kernel
    // =========================================================================
    print_sep("3. CuTe SM90: Ping-Pong (Producer WG + Consumer WG, kStage=4)");

    {
        using Cfg = PingPongConfig;

        // 复用 kernel 1 中构建好的 tma_a / tma_b
        // 重新构建一次（独立作用域，避免依赖上面的局部变量）
        auto tensor_A_pp = cute::make_tensor(
            cute::make_gmem_ptr(d_A),
            cute::make_shape(M, K),
            cute::make_stride(K, cute::_1{})
        );
        auto tensor_B_pp = cute::make_tensor(
            cute::make_gmem_ptr(d_B),
            cute::make_shape(N, K),
            cute::make_stride(K, cute::_1{})
        );
        auto smem_layout_a_pp = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomA{},
            cute::make_shape(cute::Int<Cfg::kTileM>{}, cute::Int<Cfg::kTileK>{})
        );
        auto smem_layout_b_pp = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomB{},
            cute::make_shape(cute::Int<Cfg::kTileN>{}, cute::Int<Cfg::kTileK>{})
        );
        auto tma_a_pp = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{}, tensor_A_pp, smem_layout_a_pp, cute::Int<1>{}
        );
        auto tma_b_pp = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{}, tensor_B_pp, smem_layout_b_pp, cute::Int<1>{}
        );

        constexpr size_t smem_pp = gemm_sm90::get_smem_size_pingpong<Cfg>();
        printf("SMEM: A=%zuB B=%zuB mbar=%zuB total=%zuB (%.1fKB)\n",
               cute::cosize(typename Cfg::SmemLayoutA{}) * sizeof(T),
               cute::cosize(typename Cfg::SmemLayoutB{}) * sizeof(T),
               2 * (size_t)Cfg::kStage * sizeof(uint64_t),
               smem_pp, smem_pp / 1024.0);

        if (smem_pp > prop.sharedMemPerBlockOptin) {
            printf("SKIP: Required SMEM (%zu) > device max (%zu)\n",
                   smem_pp, prop.sharedMemPerBlockOptin);
        } else {
            auto kernel_pp = gemm_sm90::gemm_kernel_pingpong<
                Cfg, decltype(tma_a_pp), decltype(tma_b_pp)>;
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel_pp,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_pp));

            dim3 block_pp(Cfg::kNumThreadsPP);  // 160 线程 (128 Consumer + 32 Producer)
            dim3 grid_pp(N / Cfg::kTileN, M / Cfg::kTileM);

            CUDA_CHECK(cudaMemset(d_C_pingpong, 0, bytes_C));
            kernel_pp<<<grid_pp, block_pp, smem_pp>>>(
                d_C_pingpong, tma_a_pp, tma_b_pp, M, N, K);
            CUDA_CHECK(cudaGetLastError());
            check_result(h_C_ref.data(), d_C_pingpong, M * N, "PingPong");

            float ms = benchmark([&](){
                kernel_pp<<<grid_pp, block_pp, smem_pp>>>(
                    d_C_pingpong, tma_a_pp, tma_b_pp, M, N, K);
            });
            printf("SM90 PingPong:      %.3f ms   %.2f TFLOPS   (vs cuBLAS: %.1f%%)\n",
                   ms, tflops(M, N, K, ms), ms_cublas / ms * 100.f);
        }
    }

    // =========================================================================
    // 3b. CuTe SM90 Ping-Pong kStage=3 (对照组, SMEM=96KB, 与单WG TMA kStage=3 对齐)
    //
    //   目的: 隔离 "SMEM 增大 -> occupancy 下降" 和 "Ping-Pong 同步开销" 两个变量
    //     若 kStage=3 Ping-Pong 仍比单WG TMA 慢 -> 说明同步/寄存器本身是瓶颈
    //     若 kStage=3 Ping-Pong ≈ 单WG TMA     -> 说明原来慢仅因 SMEM 过大
    // =========================================================================
    print_sep("3b. CuTe SM90: Ping-Pong kStage=3 (对照组, SMEM=96KB)");

    {
        using Cfg = PingPongConfig3;

        auto tensor_A_pp3 = cute::make_tensor(
            cute::make_gmem_ptr(d_A),
            cute::make_shape(M, K),
            cute::make_stride(K, cute::_1{})
        );
        auto tensor_B_pp3 = cute::make_tensor(
            cute::make_gmem_ptr(d_B),
            cute::make_shape(N, K),
            cute::make_stride(K, cute::_1{})
        );
        auto smem_layout_a_pp3 = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomA{},
            cute::make_shape(cute::Int<Cfg::kTileM>{}, cute::Int<Cfg::kTileK>{})
        );
        auto smem_layout_b_pp3 = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomB{},
            cute::make_shape(cute::Int<Cfg::kTileN>{}, cute::Int<Cfg::kTileK>{})
        );
        auto tma_a_pp3 = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{}, tensor_A_pp3, smem_layout_a_pp3, cute::Int<1>{}
        );
        auto tma_b_pp3 = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{}, tensor_B_pp3, smem_layout_b_pp3, cute::Int<1>{}
        );

        constexpr size_t smem_pp3 = gemm_sm90::get_smem_size_pingpong<Cfg>();
        printf("SMEM: A=%zuB B=%zuB mbar=%zuB total=%zuB (%.1fKB)\n",
               cute::cosize(typename Cfg::SmemLayoutA{}) * sizeof(T),
               cute::cosize(typename Cfg::SmemLayoutB{}) * sizeof(T),
               2 * (size_t)Cfg::kStage * sizeof(uint64_t),
               smem_pp3, smem_pp3 / 1024.0);

        if (smem_pp3 > prop.sharedMemPerBlockOptin) {
            printf("SKIP: Required SMEM (%zu) > device max (%zu)\n",
                   smem_pp3, prop.sharedMemPerBlockOptin);
        } else {
            auto kernel_pp3 = gemm_sm90::gemm_kernel_pingpong<
                Cfg, decltype(tma_a_pp3), decltype(tma_b_pp3)>;
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel_pp3,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_pp3));

            dim3 block_pp3(Cfg::kNumThreadsPP);  // 160 线程 (128 Consumer + 32 Producer)
            dim3 grid_pp3(N / Cfg::kTileN, M / Cfg::kTileM);

            CUDA_CHECK(cudaMemset(d_C_pingpong3, 0, bytes_C));
            kernel_pp3<<<grid_pp3, block_pp3, smem_pp3>>>(
                d_C_pingpong3, tma_a_pp3, tma_b_pp3, M, N, K);
            CUDA_CHECK(cudaGetLastError());
            check_result(h_C_ref.data(), d_C_pingpong3, M * N, "PingPong(k3)");

            float ms3 = benchmark([&](){
                kernel_pp3<<<grid_pp3, block_pp3, smem_pp3>>>(
                    d_C_pingpong3, tma_a_pp3, tma_b_pp3, M, N, K);
            });
            printf("SM90 PingPong(k3):  %.3f ms   %.2f TFLOPS   (vs cuBLAS: %.1f%%)\n",
                   ms3, tflops(M, N, K, ms3), ms_cublas / ms3 * 100.f);
        }
    }

    // =========================================================================
    // 3c. CuTe SM90 Ping-Pong kStage=2 (极限 SMEM=64KB, 理论 occupancy=3)
    //
    //   目的: 验证更高 occupancy 是否能带来更好性能
    //     SMEM=64KB → floor(227/64) = 3 blocks/SM (vs kStage=3 的 2 blocks)
    //     但 kStage=2 的流水线更短，可能增加 TMA 延迟暴露
    // =========================================================================
    print_sep("3c. CuTe SM90: Ping-Pong kStage=2 (SMEM=64KB, occupancy=3?)");

    {
        using Cfg = PingPongConfig2;

        auto tensor_A_pp2 = cute::make_tensor(
            cute::make_gmem_ptr(d_A),
            cute::make_shape(M, K),
            cute::make_stride(K, cute::_1{})
        );
        auto tensor_B_pp2 = cute::make_tensor(
            cute::make_gmem_ptr(d_B),
            cute::make_shape(N, K),
            cute::make_stride(K, cute::_1{})
        );
        auto smem_layout_a_pp2 = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomA{},
            cute::make_shape(cute::Int<Cfg::kTileM>{}, cute::Int<Cfg::kTileK>{})
        );
        auto smem_layout_b_pp2 = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomB{},
            cute::make_shape(cute::Int<Cfg::kTileN>{}, cute::Int<Cfg::kTileK>{})
        );
        auto tma_a_pp2 = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{}, tensor_A_pp2, smem_layout_a_pp2, cute::Int<1>{}
        );
        auto tma_b_pp2 = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{}, tensor_B_pp2, smem_layout_b_pp2, cute::Int<1>{}
        );

        constexpr size_t smem_pp2 = gemm_sm90::get_smem_size_pingpong<Cfg>();
        printf("SMEM: A=%zuB B=%zuB mbar=%zuB total=%zuB (%.1fKB)\n",
               cute::cosize(typename Cfg::SmemLayoutA{}) * sizeof(T),
               cute::cosize(typename Cfg::SmemLayoutB{}) * sizeof(T),
               2 * (size_t)Cfg::kStage * sizeof(uint64_t),
               smem_pp2, smem_pp2 / 1024.0);

        if (smem_pp2 > prop.sharedMemPerBlockOptin) {
            printf("SKIP: Required SMEM (%zu) > device max (%zu)\n",
                   smem_pp2, prop.sharedMemPerBlockOptin);
        } else {
            auto kernel_pp2 = gemm_sm90::gemm_kernel_pingpong<
                Cfg, decltype(tma_a_pp2), decltype(tma_b_pp2)>;
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel_pp2,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_pp2));

            dim3 block_pp2(Cfg::kNumThreadsPP);
            dim3 grid_pp2(N / Cfg::kTileN, M / Cfg::kTileM);

            CUDA_CHECK(cudaMemset(d_C_pingpong3, 0, bytes_C));  // 复用 d_C_pingpong3
            kernel_pp2<<<grid_pp2, block_pp2, smem_pp2>>>(
                d_C_pingpong3, tma_a_pp2, tma_b_pp2, M, N, K);
            CUDA_CHECK(cudaGetLastError());
            check_result(h_C_ref.data(), d_C_pingpong3, M * N, "PingPong(k2)");

            float ms2 = benchmark([&](){
                kernel_pp2<<<grid_pp2, block_pp2, smem_pp2>>>(
                    d_C_pingpong3, tma_a_pp2, tma_b_pp2, M, N, K);
            });
            printf("SM90 PingPong(k2):  %.3f ms   %.2f TFLOPS   (vs cuBLAS: %.1f%%)\n",
                   ms2, tflops(M, N, K, ms2), ms_cublas / ms2 * 100.f);
        }
    }

    // =========================================================================
    // 3d. CuTe SM90 Ping-Pong Persistent (kStage=3, 动态 tile 调度)
    //
    //   目的: 消除 wave 切换开销 (1024 tiles / 156 active blocks ≈ 6.6 waves)
    //     通过 atomicAdd 动态分配 tiles, 实现 work stealing
    //     Grid = num_sm × occupancy = 78 × 2 = 156 blocks (填满所有 SM)
    //     对比 v5 (1024 blocks), 减少 kernel launch 次数和调度开销
    // =========================================================================
    print_sep("3d. CuTe SM90: PingPong Persistent (kStage=3, 156 blocks)");

    {
        using Cfg = PingPongConfig3;  // kStage=3, 128x128x64

        auto tensor_A_pers = cute::make_tensor(
            cute::make_gmem_ptr(d_A),
            cute::make_shape(M, K),
            cute::make_stride(K, cute::_1{})
        );
        auto tensor_B_pers = cute::make_tensor(
            cute::make_gmem_ptr(d_B),
            cute::make_shape(N, K),
            cute::make_stride(K, cute::_1{})
        );
        auto smem_layout_a_pers = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomA{},
            cute::make_shape(cute::Int<Cfg::kTileM>{}, cute::Int<Cfg::kTileK>{})
        );
        auto smem_layout_b_pers = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomB{},
            cute::make_shape(cute::Int<Cfg::kTileN>{}, cute::Int<Cfg::kTileK>{})
        );
        auto tma_a_pers = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{}, tensor_A_pers, smem_layout_a_pers, cute::Int<1>{}
        );
        auto tma_b_pers = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{}, tensor_B_pers, smem_layout_b_pers, cute::Int<1>{}
        );

        constexpr size_t smem_pers = gemm_sm90::get_smem_size_pingpong_persistent<Cfg>();
        printf("SMEM: %.1fKB, grid: %d blocks (num_sm=%d × occ=2)\n",
               smem_pers / 1024.0, prop.multiProcessorCount * 2, prop.multiProcessorCount);

        if (smem_pers > prop.sharedMemPerBlockOptin) {
            printf("SKIP: Required SMEM (%zu) > device max (%zu)\n",
                   smem_pers, prop.sharedMemPerBlockOptin);
        } else {
            auto kernel_pers = gemm_sm90::gemm_kernel_pingpong_persistent<
                Cfg, decltype(tma_a_pers), decltype(tma_b_pers)>;
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel_pers,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_pers));

            // tile counter: 全局原子计数器
            int* d_tile_counter = nullptr;
            CUDA_CHECK(cudaMalloc(&d_tile_counter, sizeof(int)));

            int num_tiles_n = N / Cfg::kTileN;
            int num_tiles_m = M / Cfg::kTileM;
            int num_persistent = prop.multiProcessorCount * 2;  // 78×2=156

            dim3 block_pers(Cfg::kNumThreadsPP);
            dim3 grid_pers(num_persistent);

            // 正确性验证
            CUDA_CHECK(cudaMemset(d_C_pingpong3, 0, bytes_C));
            CUDA_CHECK(cudaMemset(d_tile_counter, 0, sizeof(int)));
            kernel_pers<<<grid_pers, block_pers, smem_pers>>>(
                d_C_pingpong3, d_tile_counter,
                tma_a_pers, tma_b_pers,
                M, N, K, num_tiles_n, num_tiles_m);
            CUDA_CHECK(cudaGetLastError());
            check_result(h_C_ref.data(), d_C_pingpong3, M * N, "PingPong-Persistent");

            // Benchmark
            float ms_pers = benchmark([&](){
                CUDA_CHECK(cudaMemset(d_tile_counter, 0, sizeof(int)));
                kernel_pers<<<grid_pers, block_pers, smem_pers>>>(
                    d_C_pingpong3, d_tile_counter,
                    tma_a_pers, tma_b_pers,
                    M, N, K, num_tiles_n, num_tiles_m);
            });
            printf("SM90 PingPong-Pers: %.3f ms   %.2f TFLOPS   (vs cuBLAS: %.1f%%)\n",
                   ms_pers, tflops(M, N, K, ms_pers), ms_cublas / ms_pers * 100.f);

            CUDA_CHECK(cudaFree(d_tile_counter));
        }
    }

    // =========================================================================
    // 3e. CuTe SM90 Ping-Pong Cluster (kStage=3, Cluster=1x2 TMA-A Multicast)
    //
    //   目的: 减少 A 的 TMA 带宽开销
    //     cluster_size=2 (N 方向): 两个 CTA 共享同一 M tile 的 A 数据
    //     TMA multicast: 每个 CTA 只加载 A 的 1/2, 通过 multicast 让另一个 CTA 也得到该部分
    //     TMA A 并发度翻倍, 减少 A 的 HBM 读取 50% (原理上)
    //
    //   __cluster_dims__(2, 1, 1): X 方向 cluster_size=2 (N 方向 2 个 CTA)
    //   grid.x = N/kTileN (实际 block 数量, 需要是 2 的倍数)
    //   grid.y = M/kTileM
    // =========================================================================
    print_sep("3e. CuTe SM90: PingPong Cluster (Cluster=1x2, TMA-A Multicast)");

    {
        using Cfg = PingPongConfig3;  // kStage=3, 128x128x64

        auto tensor_A_cl = cute::make_tensor(
            cute::make_gmem_ptr(d_A),
            cute::make_shape(M, K),
            cute::make_stride(K, cute::_1{})
        );
        auto tensor_B_cl = cute::make_tensor(
            cute::make_gmem_ptr(d_B),
            cute::make_shape(N, K),
            cute::make_stride(K, cute::_1{})
        );

        // smem layout for A (一个 CTA 的完整 A tile)
        auto smem_layout_a_cl = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomA{},
            cute::make_shape(cute::Int<Cfg::kTileM>{}, cute::Int<Cfg::kTileK>{})
        );
        // smem layout for B (普通, 一个 CTA 的 B tile)
        auto smem_layout_b_cl = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomB{},
            cute::make_shape(cute::Int<Cfg::kTileN>{}, cute::Int<Cfg::kTileK>{})
        );

        // TMA A: SM90_TMA_LOAD_MULTICAST, cluster_size=2
        //   第四个参数 cute::Int<2>{} 表示 cluster_size=2
        //   这会将 TMA box 的 M 维度缩小为 kTileM/2, 每个 CTA 只加载 A 的一半
        //   multicast_mask 在 kernel 内指定 (=0b11 for cluster_size=2)
        auto tma_a_cl = cute::make_tma_copy(
            cute::SM90_TMA_LOAD_MULTICAST{},
            tensor_A_cl,
            smem_layout_a_cl,
            cute::Int<2>{}   // cluster_size (multicast 的 CTA 数量)
        );
        // TMA B: 普通 SM90_TMA_LOAD
        auto tma_b_cl = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{},
            tensor_B_cl,
            smem_layout_b_cl,
            cute::Int<1>{}
        );

        constexpr size_t smem_cl = gemm_sm90::get_smem_size_pingpong_cluster<Cfg>();
        printf("SMEM: %.1fKB, cluster_dims=(2,1,1), grid=(%d,%d)\n",
               smem_cl / 1024.0, N / Cfg::kTileN, M / Cfg::kTileM);

        if (smem_cl > prop.sharedMemPerBlockOptin) {
            printf("SKIP: Required SMEM (%zu) > device max (%zu)\n",
                   smem_cl, prop.sharedMemPerBlockOptin);
        } else {
            auto kernel_cl = gemm_sm90::gemm_kernel_pingpong_cluster<
                Cfg, decltype(tma_a_cl), decltype(tma_b_cl)>;
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel_cl,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_cl));

            dim3 block_cl(Cfg::kNumThreadsPP);
            // grid.x = N/kTileN (必须是 cluster_size=2 的倍数)
            // grid.y = M/kTileM
            int gx = N / Cfg::kTileN;
            int gy = M / Cfg::kTileM;
            dim3 grid_cl(gx, gy);

            // 使用 cudaLaunchKernelExArgs 指定 cluster 形状
            // (或直接用 __cluster_dims__ 编译时指定)
            cudaLaunchConfig_t launch_config = {};
            launch_config.gridDim  = grid_cl;
            launch_config.blockDim = block_cl;
            launch_config.dynamicSmemBytes = smem_cl;
            launch_config.stream   = nullptr;

            cudaLaunchAttribute attr[1];
            attr[0].id = cudaLaunchAttributeClusterDimension;
            attr[0].val.clusterDim = {2, 1, 1};
            launch_config.numAttrs = 1;
            launch_config.attrs    = attr;

            // 使用 cudaLaunchKernelEx 直接传参 (模板版本)
            auto launch_cluster = [&](float* Cptr) {
                return cudaLaunchKernelEx(&launch_config,
                                          kernel_cl,
                                          Cptr, tma_a_cl, tma_b_cl, M, N, K);
            };

            // 正确性验证
            CUDA_CHECK(cudaMemset(d_C_pingpong3, 0, bytes_C));
            CUDA_CHECK(launch_cluster(d_C_pingpong3));
            CUDA_CHECK(cudaGetLastError());
            check_result(h_C_ref.data(), d_C_pingpong3, M * N, "PingPong-Cluster");

            // Benchmark
            float ms_cl = benchmark([&](){
                CUDA_CHECK(launch_cluster(d_C_pingpong3));
            });
            printf("SM90 PingPong-Cluster: %.3f ms   %.2f TFLOPS   (vs cuBLAS: %.1f%%)\n",
                   ms_cl, tflops(M, N, K, ms_cl), ms_cublas / ms_cl * 100.f);
        }
    }

    // =========================================================================
    // 4. CuTe SM90 Ping-Pong + TMA Store Epilogue (版本 A, 160 线程)
    //
    //   基于 hpc-ops 学习的 TMA Store 优化:
    //   - Epilogue: FP32 累加器 → float SMEM → TMA Store (FP32)
    //   - 输出类型: FP32
    //   - TMA Store 是异步的, 不阻塞 Math WG
    //   - SMEM (kStage=2): A(32) + B(32) + C_float(64) + mbar ≈ 128KB ✓
    // =========================================================================
    print_sep("4. CuTe SM90: PingPong v6 + TMA Store Epilogue (160 线程, FP32 out)");

    {
        using Cfg = PPTmaStoreConfig;

        auto tensor_A_v6 = cute::make_tensor(
            cute::make_gmem_ptr(d_A),
            cute::make_shape(M, K), cute::make_stride(K, cute::_1{}));
        auto tensor_B_v6 = cute::make_tensor(
            cute::make_gmem_ptr(d_B),
            cute::make_shape(N, K), cute::make_stride(K, cute::_1{}));
        // TMA Store C (FP32 output, row-major): shape (M, N)
        auto tensor_C_v6 = cute::make_tensor(
            cute::make_gmem_ptr(d_C_v6),
            cute::make_shape(M, N), cute::make_stride(N, cute::_1{}));

        auto smem_layout_a_v6 = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomA{},
            cute::make_shape(cute::Int<Cfg::kTileM>{}, cute::Int<Cfg::kTileK>{}));
        auto smem_layout_b_v6 = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomB{},
            cute::make_shape(cute::Int<Cfg::kTileN>{}, cute::Int<Cfg::kTileK>{}));
        // C buffer SMEM layout for TMA Store: float row-major (M, N) stride (N, 1)
        // TMA Store 不要求 swizzle, row-major 最简单
        auto smem_layout_c_v6 = cute::make_layout(
            cute::make_shape(cute::Int<Cfg::kTileM>{}, cute::Int<Cfg::kTileN>{}),
            cute::make_stride(cute::Int<Cfg::kTileN>{}, cute::_1{}));

        auto tma_a_v6 = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{}, tensor_A_v6, smem_layout_a_v6, cute::Int<1>{});
        auto tma_b_v6 = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{}, tensor_B_v6, smem_layout_b_v6, cute::Int<1>{});
        // TMA Store: float SMEM → float GMEM
        auto tma_c_v6 = cute::make_tma_copy(
            cute::SM90_TMA_STORE{}, tensor_C_v6, smem_layout_c_v6);

        constexpr size_t smem_v6 = gemm_sm90::get_smem_size_pingpong_tma_store<Cfg>();
        printf("SMEM: A=%zuKB B=%zuKB C_buf=%zuKB (float row-major) total=%.1fKB\n",
               cute::cosize(typename Cfg::SmemLayoutA{}) * sizeof(T) / 1024,
               cute::cosize(typename Cfg::SmemLayoutB{}) * sizeof(T) / 1024,
               (size_t)Cfg::kTileM * Cfg::kTileN * sizeof(float) / 1024,
               smem_v6 / 1024.0);

        if (smem_v6 > prop.sharedMemPerBlockOptin) {
            printf("SKIP: Required SMEM (%zu) > device max (%zu)\n",
                   smem_v6, prop.sharedMemPerBlockOptin);
        } else {
            auto kernel_v6 = gemm_sm90::gemm_kernel_pingpong_tma_store<
                Cfg, decltype(tma_a_v6), decltype(tma_b_v6), decltype(tma_c_v6)>;
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel_v6, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_v6));

            dim3 block_v6(Cfg::kNumThreadsPP);  // 160 线程
            dim3 grid_v6(N / Cfg::kTileN, M / Cfg::kTileM);

            // 正确性验证
            CUDA_CHECK(cudaMemset(d_C_v6, 0, bytes_C));
            kernel_v6<<<grid_v6, block_v6, smem_v6>>>(
                tma_c_v6, tma_a_v6, tma_b_v6, M, N, K);
            CUDA_CHECK(cudaGetLastError());
            check_result(h_C_ref.data(), d_C_v6, M * N, "PP-TMAStore-v6");

            // Performance
            float ms_v6 = benchmark([&](){
                kernel_v6<<<grid_v6, block_v6, smem_v6>>>(
                    tma_c_v6, tma_a_v6, tma_b_v6, M, N, K);
            });
            printf("SM90 PP-TMAStore-v6: %.3f ms   %.2f TFLOPS   (vs cuBLAS: %.1f%%)\n",
                   ms_v6, tflops(M, N, K, ms_v6), ms_cublas / ms_v6 * 100.f);
        }
    }

    // =========================================================================
    // 5. CuTe SM90 双 WG Cooperative GEMM (版本 B, 384 线程)
    //
    //   基于 hpc-ops 学习的双 WG 优化:
    //   - 384 线程 = 2 Math WG (tid 0-255) + 1 Load WG (tid 256-383)
    //   - 每个 block 同时处理 2 个相邻 N tile (grid.x 减半)
    //   - warpgroup_reg_alloc<168> (Math WG) + warpgroup_reg_dealloc<24> (Load WG)
    //   - Epilogue: FP32 直接 STG 写回 GMEM (无 SMEM C buffer)
    //   - SMEM (kStage=2): A(32) + B0(32) + B1(32) + mbar ≈ 96KB ✓
    // =========================================================================
    print_sep("5. CuTe SM90: 双WG Cooperative (384 线程, FP32 out)");

    {
        using Cfg = Config2WG;

        auto tensor_A_2wg = cute::make_tensor(
            cute::make_gmem_ptr(d_A),
            cute::make_shape(M, K), cute::make_stride(K, cute::_1{}));
        auto tensor_B_2wg = cute::make_tensor(
            cute::make_gmem_ptr(d_B),
            cute::make_shape(N, K), cute::make_stride(K, cute::_1{}));

        auto smem_layout_a_2wg = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomA{},
            cute::make_shape(cute::Int<Cfg::kTileM>{}, cute::Int<Cfg::kTileK>{}));
        auto smem_layout_b_2wg = cute::tile_to_shape(
            typename Cfg::SmemLayoutAtomB{},
            cute::make_shape(cute::Int<Cfg::kTileN>{}, cute::Int<Cfg::kTileK>{}));

        auto tma_a_2wg = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{}, tensor_A_2wg, smem_layout_a_2wg, cute::Int<1>{});
        auto tma_b_2wg = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{}, tensor_B_2wg, smem_layout_b_2wg, cute::Int<1>{});

        constexpr size_t smem_2wg = gemm_sm90::get_smem_size_2wg_pingpong<Cfg>();
        printf("SMEM: A=%zuKB B0=%zuKB B1=%zuKB mbar total=%.1fKB\n",
               cute::cosize(typename Cfg::SmemLayoutA{}) * sizeof(T) / 1024,
               cute::cosize(typename Cfg::SmemLayoutB{}) * sizeof(T) / 1024,
               cute::cosize(typename Cfg::SmemLayoutB{}) * sizeof(T) / 1024,
               smem_2wg / 1024.0);

        // 检查 N 是否能被 kTileN*2 整除 (版本 B 要求 grid.x = N/(kTileN*2))
        if (N % (Cfg::kTileN * 2) != 0) {
            printf("SKIP: N=%d not divisible by kTileN*2=%d\n",
                   N, Cfg::kTileN * 2);
        } else if (smem_2wg > prop.sharedMemPerBlockOptin) {
            printf("SKIP: Required SMEM (%zu) > device max (%zu)\n",
                   smem_2wg, prop.sharedMemPerBlockOptin);
        } else {
            auto kernel_2wg = gemm_sm90::gemm_kernel_2wg_pingpong<
                Cfg, decltype(tma_a_2wg), decltype(tma_b_2wg)>;
            CUDA_CHECK(cudaFuncSetAttribute(
                kernel_2wg, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_2wg));

            dim3 block_2wg(Cfg::kNumThreads2WG);  // 384 线程
            // grid.x 减半: 每 block 处理 2 个 N tile
            dim3 grid_2wg(N / (Cfg::kTileN * 2), M / Cfg::kTileM);

            // 正确性验证
            CUDA_CHECK(cudaMemset(d_C_2wg, 0, bytes_C));
            kernel_2wg<<<grid_2wg, block_2wg, smem_2wg>>>(
                d_C_2wg, tma_a_2wg, tma_b_2wg, M, N, K);
            CUDA_CHECK(cudaGetLastError());
            check_result(h_C_ref.data(), d_C_2wg, M * N, "2WG-Coop");

            // Performance
            float ms_2wg = benchmark([&](){
                kernel_2wg<<<grid_2wg, block_2wg, smem_2wg>>>(
                    d_C_2wg, tma_a_2wg, tma_b_2wg, M, N, K);
            });
            printf("SM90 2WG-Coop:       %.3f ms   %.2f TFLOPS   (vs cuBLAS: %.1f%%)\n",
                   ms_2wg, tflops(M, N, K, ms_2wg), ms_cublas / ms_2wg * 100.f);
        }
    }

    // =========================================================================
    // 6. 性能汇总
    // =========================================================================
    print_sep("Performance Summary");
    printf("  M=%d  N=%d  K=%d  (BF16 -> FP32)\n", M, N, K);
    printf("  Peak BF16 Tensor Core (theoretical): %.0f TFLOPS\n",
           (double)prop.clockRate * 1e-6 *
           (double)prop.multiProcessorCount * 512.0 * 1e-3);
    printf("\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_cublas));
    CUDA_CHECK(cudaFree(d_C_tma));
    CUDA_CHECK(cudaFree(d_C_cpasync));
    CUDA_CHECK(cudaFree(d_C_pingpong));
    CUDA_CHECK(cudaFree(d_C_pingpong3));
    CUDA_CHECK(cudaFree(d_C_v6));
    CUDA_CHECK(cudaFree(d_C_2wg));
    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}
