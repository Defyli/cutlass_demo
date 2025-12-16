#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <functional>

#include "utils.cuh"
#include "gemm_simple.cuh"
#include "gemm_double_buffer.cuh"
#include "gemm_multi_stage.cuh"
#include "gemm_opt_final.cuh"
#include "gemm_boundary_check.cuh"

using T = cute::half_t;

void run_cublas(cublasHandle_t handle, T* d_A, T* d_B, T* d_C, int M, int N, int K) {
    half alpha = 1.0f;
    half beta = 0.0f;
    
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K, 
            &alpha,
            (half *)d_B, K,
            (half *)d_A, K,
            &beta,
            (half *)d_C, N);
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
    // 你可以修改这里的 M, N, K 来测试非对齐的情况，例如 4097, 4099, 2050
    int M = 5000;
    int N = 5000;
    int K = 2048;

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
    using ConfigSimple = gemm_simple::GemmConfig<T, 128, 128, 32, 128>;
    int smem_simple = sizeof(T) * (cute::size(typename ConfigSimple::SmemLayoutA{}) + cute::size(typename ConfigSimple::SmemLayoutB{}));
    dim3 grid(N / 128, M / 128);
    dim3 block(128);
    
    if (M % 128 != 0 || N % 128 != 0 || K % 32 != 0) {
        printf("Simple: \tSkipped (Dimensions not aligned with Tile size)\n");
    } else {
        CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
        gemm_simple::gemm_kernel<ConfigSimple><<<grid, block, smem_simple>>>(d_C, d_A, d_B, M, N, K);
        check_result(h_C_ref, d_C, M*N, "Simple");

        float ms_simple = benchmark([&](){ 
            gemm_simple::gemm_kernel<ConfigSimple><<<grid, block, smem_simple>>>(d_C, d_A, d_B, M, N, K); 
        }, 20);
        printf("Simple: \t%.3f ms \t%.2f TFLOPS\n", ms_simple, tflops / (ms_simple * 1e-3));
    }

    // 3. Run Double Buffer
    using ConfigDouble = gemm_double_buffer::GemmConfig<T, 128, 128, 32,128>;
    int smem_double = sizeof(T) * (cute::size(typename ConfigDouble::SmemLayoutA{}) + cute::size(typename ConfigDouble::SmemLayoutB{}));
    
    if (M % 128 != 0 || N % 128 != 0 || K % 32 != 0) {
        printf("DoubleBuf: \tSkipped (Dimensions not aligned with Tile size)\n");
    } else {
        CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
        gemm_double_buffer::gemm_kernel<ConfigDouble><<<grid, block, smem_double>>>(d_C, d_A, d_B, M, N, K);
        check_result(h_C_ref, d_C, M*N, "DoubleBuffer");

        float ms_double = benchmark([&](){ 
            gemm_double_buffer::gemm_kernel<ConfigDouble><<<grid, block, smem_double>>>(d_C, d_A, d_B, M, N, K); 
        }, 20);
        printf("DoubleBuf: \t%.3f ms \t%.2f TFLOPS\n", ms_double, tflops / (ms_double * 1e-3));
    }

    // 4. Run Multi Stage
    using ConfigMulti = gemm_multi_stage::GemmConfig<T, 128, 128, 32, 3, 128>; // 3 Stages
    int smem_multi = sizeof(T) * (cute::size(typename ConfigMulti::SmemLayoutA{}) + cute::size(typename ConfigMulti::SmemLayoutB{}));
    
    // Set dynamic shared memory limit
    CUDA_CHECK(cudaFuncSetAttribute(gemm_multi_stage::gemm_kernel<ConfigMulti>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_multi));

    if (M % 128 != 0 || N % 128 != 0 || K % 32 != 0) {
        printf("MultiStage: \tSkipped (Dimensions not aligned with Tile size)\n");
    } else {
        CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
        gemm_multi_stage::gemm_kernel<ConfigMulti><<<grid, block, smem_multi>>>(d_C, d_A, d_B, M, N, K);
        check_result(h_C_ref, d_C, M*N, "MultiStage");

        float ms_multi = benchmark([&](){ 
            gemm_multi_stage::gemm_kernel<ConfigMulti><<<grid, block, smem_multi>>>(d_C, d_A, d_B, M, N, K); 
        }, 20);
        printf("MultiStage: \t%.3f ms \t%.2f TFLOPS\n", ms_multi, tflops / (ms_multi * 1e-3));
    }

    // 5. Run Final Opt
    using ConfigFinal = gemm_final_opt::GemmConfig<T, 128, 128, 32, 3, 128>;
    int smem_final = sizeof(T) * (cute::size(typename ConfigFinal::SmemLayoutA{}) + cute::size(typename ConfigFinal::SmemLayoutB{}));

    CUDA_CHECK(cudaFuncSetAttribute(gemm_final_opt::gemm_kernel<ConfigFinal>,cudaFuncAttributeMaxDynamicSharedMemorySize, smem_final));

    if (M % 128 != 0 || N % 128 != 0 || K % 32 != 0) {
        printf("Final opt: \tSkipped (Dimensions not aligned with Tile size)\n");
    } else {
        CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
        gemm_final_opt::gemm_kernel<ConfigFinal><<<grid, block, smem_final>>>(d_C, d_A, d_B, M, N, K);
        check_result(h_C_ref, d_C, M*N, "Final Opt");

        float ms_final = benchmark([&](){
            gemm_final_opt::gemm_kernel<ConfigFinal><<<grid, block, smem_final>>>(d_C, d_A, d_B, M, N, K);
        }, 20);
        printf("Final opt: \t%.3f ms \t%.2f TFLOPS\n", ms_final, tflops / (ms_final * 1e-3));
    }

    // 6. Run Boundary Check
    using ConfigCheck = gemm_boundary_check::GemmConfig<T, 128, 128, 32, 3, 128>;
    int smem_check = sizeof(T) * (cute::size(typename ConfigCheck::SmemLayoutA{}) + cute::size(typename ConfigCheck::SmemLayoutB{}));

    CUDA_CHECK(cudaFuncSetAttribute(gemm_boundary_check::gemm_kernel<ConfigCheck>,cudaFuncAttributeMaxDynamicSharedMemorySize, smem_check));

    // 对于 Boundary Check 版本，Grid 需要向上取整以覆盖边缘
    dim3 grid_check((N + 128 - 1) / 128, (M + 128 - 1) / 128);

    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
    gemm_boundary_check::gemm_kernel<ConfigCheck><<<grid_check, block, smem_check>>>(d_C, d_A, d_B, M, N, K);
    check_result(h_C_ref, d_C, M*N, "Boundary Check");

    float ms_check = benchmark([&](){
        gemm_boundary_check::gemm_kernel<ConfigCheck><<<grid_check, block, smem_check>>>(d_C, d_A, d_B, M, N, K);
    }, 20);
    printf("Boundary Check: \t%.3f ms \t%.2f TFLOPS\n", ms_check, tflops / (ms_check * 1e-3));

    // Cleanup
    free(h_A); free(h_B); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}