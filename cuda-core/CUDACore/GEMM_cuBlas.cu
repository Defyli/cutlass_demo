#include <stdint.h>
#include<cublas_v2.h>
#include<mma.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cstdio>
#include "Kernel.h"

void HGEMM_CHECK_CUBLAS_ERROR(cublasStatus_t e)
{
    if(e!=CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS ERROR");
        exit(EXIT_FAILURE); 
    }
}

cublasHandle_t getCublasTensorOpHandle() {
    cublasHandle_t handle = nullptr;
    HGEMM_CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    HGEMM_CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    return handle;
}



void CUDA_Kernel::cublasTensorOp(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    static cublasHandle_t handle = getCublasTensorOpHandle();
    static half alpha = 1.0;
    static half beta = 0.0;

    half*d_a,*d_b,*d_c;
    cudaMallocAsync(&d_a,sizeof(half)*M*K,0);
    cudaMallocAsync(&d_b,sizeof(half)*N*K,0);
    cudaMallocAsync(&d_c,sizeof(half)*N*M,0);

    cudaMemcpyAsync(d_a,A,sizeof(half)*M*K,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_b,B,sizeof(half)*N*K,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_c,C,sizeof(half)*M*N,cudaMemcpyHostToDevice,0);

    HGEMM_CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, d_b, CUDA_R_16F, K, d_a,
                                          CUDA_R_16F, K, &beta, d_c, CUDA_R_16F, N, CUBLAS_COMPUTE_16F,
                                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    
    cudaMemcpyAsync(C,d_c,sizeof(half)*M*N,cudaMemcpyDeviceToHost,0);
    cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
    cudaStreamSynchronize(0);

}
