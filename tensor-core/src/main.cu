#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <functional>

#include "utils.cuh"
#include "gemm_simple.cuh"
#include "gemm_double_buffer.cuh"
#include "gemm_multi_stage.cuh"

using T = cute::half_t;

void run_cublas(cublasHandle_t handle, T* d_A, T* d_B, T* d_C, int M, int N, int K) {
    half alpha = 1.0f;
    half beta = 0.0f;
    // 注意：cuBLAS 是列主序，我们这里假设 A, B 已经是适应 cuBLAS 的布局
    // 或者我们使用 cublasGemmEx 并指定 OP_T/OP_N 来适配
    // 这里的逻辑假设 A(M,K) RowMajor, B(N,K) RowMajor (即 B^T ColMajor)
    // C = A * B^T
    cublasStatus_t ret = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_16F, K,
                 d_A, CUDA_R_16F, K,
                 &beta,
                 d_C, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (ret != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS Error: %d\n", ret);
    }
}

template<typename KernelFunc>
float benchmark(KernelFunc func, int n_iter) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for(int i=0; i<5; ++i) func();

    cudaEventRecord(start);
    for(int i=0; i<n_iter; ++i) func();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / n_iter;
}

int main() {
    int M = 4096;
    int N = 4096;
    int K = 4096;

    printf("Benchmarking GEMM FP16: M=%d, N=%d, K=%d\n", M, N, K);

    size_t bytes_A = M * K * sizeof(T);
    size_t bytes_B = N * K * sizeof(T);
    size_t bytes_C = M * N * sizeof(T);

    T *h_A = (T*)malloc(bytes_A);
    T *h_B = (T*)malloc(bytes_B);
    T *h_C_ref = (T*)malloc(bytes_C); // Host reference (from cuBLAS)

    gen_rand_data(h_A, M * K);
    gen_rand_data(h_B, N * K);

    T *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // 1. Run cuBLAS (Baseline & Correctness Reference)
    run_cublas(handle, d_A, d_B, d_C, M, N, K); // Warmup & Result
    CUDA_CHECK(cudaMemcpy(h_C_ref, d_C, bytes_C, cudaMemcpyDeviceToHost));
    
    float ms_cublas = benchmark([&](){ run_cublas(handle, d_A, d_B, d_C, M, N, K); }, 20);
    double tflops = (2.0 * M * N * K) * 1e-12;
    printf("cuBLAS: \t%.3f ms \t%.2f TFLOPS\n", ms_cublas, tflops / (ms_cublas * 1e-3));

    // 2. Run Simple
    using ConfigSimple = gemm_simple::GemmConfig<T, 128, 128, 32>;
    int smem_simple = sizeof(T) * (cute::cosize(typename ConfigSimple::SmemLayoutA{}) + cute::cosize(typename ConfigSimple::SmemLayoutB{}));
    dim3 grid(N / 128, M / 128);
    dim3 block(128);
    
    // Check correctness
    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
    gemm_simple::gemm_kernel<ConfigSimple><<<grid, block, smem_simple>>>(d_C, d_A, d_B, M, N, K);
    check_result(h_C_ref, d_C, M*N, "Simple");

    float ms_simple = benchmark([&](){ 
        gemm_simple::gemm_kernel<ConfigSimple><<<grid, block, smem_simple>>>(d_C, d_A, d_B, M, N, K); 
    }, 20);
    printf("Simple: \t%.3f ms \t%.2f TFLOPS\n", ms_simple, tflops / (ms_simple * 1e-3));

    // 3. Run Double Buffer
    using ConfigDouble = gemm_double_buffer::GemmConfig<T, 128, 128, 32>;
    int smem_double = sizeof(T) * (cute::cosize(typename ConfigDouble::SmemLayoutA{}) + cute::cosize(typename ConfigDouble::SmemLayoutB{}));
    
    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
    gemm_double_buffer::gemm_kernel<ConfigDouble><<<grid, block, smem_double>>>(d_C, d_A, d_B, M, N, K);
    check_result(h_C_ref, d_C, M*N, "DoubleBuffer");

    float ms_double = benchmark([&](){ 
        gemm_double_buffer::gemm_kernel<ConfigDouble><<<grid, block, smem_double>>>(d_C, d_A, d_B, M, N, K); 
    }, 20);
    printf("DoubleBuf: \t%.3f ms \t%.2f TFLOPS\n", ms_double, tflops / (ms_double * 1e-3));

    // 4. Run Multi Stage
    using ConfigMulti = gemm_multi_stage::GemmConfig<T, 128, 128, 32, 3>; // 3 Stages
    int smem_multi = sizeof(T) * (cute::cosize(typename ConfigMulti::SmemLayoutA{}) + cute::cosize(typename ConfigMulti::SmemLayoutB{}));
    
    // Set dynamic shared memory limit
    CUDA_CHECK(cudaFuncSetAttribute(gemm_multi_stage::gemm_kernel<ConfigMulti>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_multi));

    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
    gemm_multi_stage::gemm_kernel<ConfigMulti><<<grid, block, smem_multi>>>(d_C, d_A, d_B, M, N, K);
    check_result(h_C_ref, d_C, M*N, "MultiStage");

    float ms_multi = benchmark([&](){ 
        gemm_multi_stage::gemm_kernel<ConfigMulti><<<grid, block, smem_multi>>>(d_C, d_A, d_B, M, N, K); 
    }, 20);
    printf("MultiStage: \t%.3f ms \t%.2f TFLOPS\n", ms_multi, tflops / (ms_multi * 1e-3));

    // Cleanup
    free(h_A); free(h_B); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}