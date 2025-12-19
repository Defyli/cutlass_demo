#include <cuda_runtime.h>
#include <cublasLt.h> 
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cublas_v2.h> 
#include "gemm_fp8.cuh" 

// 定义类型别名
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = float; // 累加和输出使用 float

// 辅助函数：生成随机 FP8 数据
void gen_rand_data_fp8(ElementA* data, size_t count) {
    // 使用随机数引擎，生成范围在 [-0.5, 0.5] 的浮点数
    // 包含负数可以减少累加和的膨胀，从而减少 FP32 精度溢出
    std::default_random_engine generator(42);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);

    for (size_t i = 0; i < count; ++i) {
        float r = distribution(generator);
        data[i] = static_cast<ElementA>(r);
    }
}

// 辅助函数：检查结果
void check_result_fp8(float* host_ref, ElementC* device_res, int N, const char* name) {
    float* h_res = (float*)malloc(N * sizeof(float));
    cudaMemcpy(h_res, device_res, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    double max_diff = 0.0;
    double max_val = 0.0;
    
    for(int i=0; i<N; ++i) {
        double val = std::abs(host_ref[i]);
        if (val > max_val) max_val = val;

        double diff = std::abs(host_ref[i] - h_res[i]);
        if(diff > max_diff) max_diff = diff;
    }
    
    double rel_err = (max_val > 1e-5) ? (max_diff / max_val) : max_diff;
    
    printf("%s -> Max Diff: %f, Max Val: %f, Rel Err: %e\n", name, max_diff, max_val, rel_err);
    
    // 经验法则：FP8 Tensor Core 累加结果与 FP32 Reference 的相对误差在 1e-4 ~ 1e-3 都是正常的
    if (rel_err < 1e-3) {
        printf(">> SUCCESS: Error is within acceptable FP8 numerical drift.\n");
    } else {
        printf(">> WARNING: Error might be too high. But this may be caused by bigger K due to accumlator error\n");
    }

    free(h_res);
}

// 运行 FP32 cuBLAS 作为 Ground Truth
// 计算 C = A * B^T
void run_cublas_fp32_ref(cublasHandle_t handle, 
                         ElementA* h_A, ElementB* h_B, float* h_C_ref, 
                         int M, int N, int K) 
{
    // 1. 将 FP8 数据转为 FP32
    float* h_A_f32 = (float*)malloc(M * K * sizeof(float));
    float* h_B_f32 = (float*)malloc(N * K * sizeof(float));
    
    for(int i=0; i<M*K; ++i) h_A_f32[i] = static_cast<float>(h_A[i]);
    for(int i=0; i<N*K; ++i) h_B_f32[i] = static_cast<float>(h_B[i]);

    float *d_A_f32, *d_B_f32, *d_C_f32;
    cudaMalloc(&d_A_f32, M * K * sizeof(float));
    cudaMalloc(&d_B_f32, N * K * sizeof(float));
    cudaMalloc(&d_C_f32, M * N * sizeof(float));

    cudaMemcpy(d_A_f32, h_A_f32, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_f32, h_B_f32, N * K * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;
    
    // 逻辑说明：
    // 我们要计算 Row Major 的 C = A * B^T
    // A (MxK), B (NxK)
    // cuBLAS 是列主序 (Column Major)。
    // 传入 Row Major 的矩阵指针，cuBLAS 会将其视为转置后的 Col Major 矩阵。
    // d_B (NxK Row) -> 视为 KxN Col (即 B^T_row)
    // d_A (MxK Row) -> 视为 KxM Col (即 A^T_row)
    // 我们希望得到 C (MxN Row) -> 视为 NxM Col (即 C^T_row)
    // 公式：C^T = (A * B^T)^T = B * A^T
    // 映射到 cuBLAS Sgemm:
    // OP(B_view) * OP(A_view)
    // B_view 是 KxN. 我们需要 B (NxK). 所以用 OP_T (转置 B_view) -> NxK
    // A_view 是 KxM. 我们需要 A^T (KxM). 所以用 OP_N (不转置 A_view) -> KxM
    // 结果维度: (NxK) * (KxM) = NxM. 符合 C_view (NxM).
    
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                N, M, K, 
                &alpha, 
                d_B_f32, K, // ldb (B_view 的 leading dimension, 即 K)
                d_A_f32, K, // lda (A_view 的 leading dimension, 即 K)
                &beta, 
                d_C_f32, N); // ldc (C_view 的 leading dimension, 即 N)

    cudaMemcpy(h_C_ref, d_C_f32, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    free(h_A_f32); free(h_B_f32);
    cudaFree(d_A_f32); cudaFree(d_B_f32); cudaFree(d_C_f32);
}

// 封装 cuBLASLt FP8 GEMM
void run_cublasLt_fp8(cublasLtHandle_t ltHandle, 
                      ElementA* d_A, ElementB* d_B, ElementC* d_C, 
                      int M, int N, int K) 
{
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    
    cudaDataType_t Atype = CUDA_R_8F_E4M3;
    cudaDataType_t Btype = CUDA_R_8F_E4M3;
    cudaDataType_t Ctype = CUDA_R_32F;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;

    // C = A * B^T
    cublasOperation_t transa = CUBLAS_OP_N; 
    cublasOperation_t transb = CUBLAS_OP_T; 

    cublasLtMatmulDescCreate(&operationDesc, computeType, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t));

    // 显式设置 Layout 为 Row Major，简化维度理解
    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;

    cublasLtMatrixLayoutCreate(&Adesc, Atype, M, K, K); 
    cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
    
    cublasLtMatrixLayoutCreate(&Bdesc, Btype, N, K, K); 
    cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
    
    cublasLtMatrixLayoutCreate(&Cdesc, Ctype, M, N, N); 
    cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulPreferenceCreate(&preference);
    size_t workspaceSize = 32 * 1024 * 1024;
    void* d_workspace;
    cudaMalloc(&d_workspace, workspaceSize);
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults);

    if (returnedResults == 0) {
        printf("cuBLASLt: No valid algorithm found!\n");
        cudaFree(d_workspace);
        return;
    }

    cublasLtMatmul(ltHandle, operationDesc, 
                   &alpha, d_A, Adesc, 
                   d_B, Bdesc, 
                   &beta, d_C, Cdesc, 
                   d_C, Cdesc, 
                   &heuristicResult.algo, d_workspace, workspaceSize, 0);

    cudaFree(d_workspace);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(operationDesc);
}

template<typename KernelFunc>
float benchmark(KernelFunc func, int n_iter) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(int i=0; i<5; ++i) func(); // Warmup
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
    // 确保在 SM89 (RTX 4090 / L40) 或 SM90 (H100) 上运行
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8 || (prop.major == 8 && prop.minor < 9)) {
        printf("Error: FP8 requires SM89 or later (Ada Lovelace/Hopper). Current: SM%d%d\n", prop.major, prop.minor);
        return -1;
    }

    int M = 4096;
    int N = 4096;
    int K = 1024;

    printf("Benchmarking FP8 GEMM (E4M3): M=%d, N=%d, K=%d\n", M, N, K);

    size_t bytes_A = M * K * sizeof(ElementA);
    size_t bytes_B = N * K * sizeof(ElementB);
    size_t bytes_C = M * N * sizeof(ElementC);

    ElementA *h_A = (ElementA*)malloc(bytes_A);
    ElementB *h_B = (ElementB*)malloc(bytes_B);
    ElementC *h_C_ref = (ElementC*)malloc(bytes_C);

    gen_rand_data_fp8(h_A, M * K);
    gen_rand_data_fp8(h_B, N * K);

    ElementA *d_A, *d_B;
    ElementC *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    // 0. Run FP32 Reference (Ground Truth)
    printf("Running FP32 Reference...\n");
    float* h_C_fp32_ref = (float*)malloc(bytes_C);
    cublasHandle_t handle;
    cublasCreate(&handle);
    run_cublas_fp32_ref(handle, h_A, h_B, h_C_fp32_ref, M, N, K);
    cublasDestroy(handle);

    // 1. Run cuBLASLt (Baseline)
    printf("Running cuBLASLt FP8...\n");
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);
    
    // Warmup & Correctness check
    run_cublasLt_fp8(ltHandle, d_A, d_B, d_C, M, N, K);
    check_result_fp8(h_C_fp32_ref, d_C, M*N, "cuBLASLt FP8 vs FP32");

    float ms_cublas = benchmark([&](){ 
        run_cublasLt_fp8(ltHandle, d_A, d_B, d_C, M, N, K); 
    }, 20);
    
    // FP8 TFLOPS = 2 * M * N * K / time
    double tflops = (2.0 * M * N * K) * 1e-12;
    printf("cuBLASLt FP8: \t%.3f ms \t%.2f TFLOPS\n", ms_cublas, tflops / (ms_cublas * 1e-3));

    // 2. Run Custom CuTe FP8 Kernel
    printf("Running CuTe FP8 Kernel...\n");
    // Config: 128x128 Tile, K=64 (FP8 aligned), 3 Stages, 128 Threads
    using ConfigFP8 = gemm_fp8::GemmConfig<ElementA, ElementB, ElementC, 128, 128, 64, 3, 128>;
    
    int smem_size_ab = sizeof(ElementA) * cute::size(typename ConfigFP8::SmemLayoutA{}) + 
                       sizeof(ElementB) * cute::size(typename ConfigFP8::SmemLayoutB{});
    int smem_size_c = sizeof(ElementC) * cute::size(typename ConfigFP8::SmemLayoutC{});
    int smem_size = std::max(smem_size_ab, smem_size_c);
    
    cudaFuncSetAttribute(gemm_fp8::gemm_kernel<ConfigFP8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    dim3 block(128);
    dim3 grid(N / 128, M / 128);

    cudaMemset(d_C, 0, bytes_C);
    gemm_fp8::gemm_kernel<ConfigFP8><<<grid, block, smem_size>>>(d_C, d_A, d_B, M, N, K);
    
    check_result_fp8(h_C_fp32_ref, d_C, M*N, "CuTe FP8 vs FP32"); 

    float ms_cute = benchmark([&](){
        gemm_fp8::gemm_kernel<ConfigFP8><<<grid, block, smem_size>>>(d_C, d_A, d_B, M, N, K);
    }, 20);

    printf("CuTe FP8 Opt: \t%.3f ms \t%.2f TFLOPS\n", ms_cute, tflops / (ms_cute * 1e-3));

    // Cleanup
    free(h_A); free(h_B); free(h_C_ref); free(h_C_fp32_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasLtDestroy(ltHandle);

    return 0;
}