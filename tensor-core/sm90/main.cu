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

// 验证结果 (比对 device 结果与 host reference)
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

    CUDA_CHECK(cudaMalloc(&d_A,          bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B,          bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C_cublas,   bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_tma,      bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_cpasync,  bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_pingpong, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_pingpong3,bytes_C));

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

            dim3 block_pp(Cfg::kNumThreadsPP);  // 256 线程
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

            dim3 block_pp3(Cfg::kNumThreadsPP);  // 256 线程
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
    // 4. 性能汇总
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
    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}
