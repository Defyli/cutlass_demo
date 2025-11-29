#ifndef KERNEL_H
#define KERNEL_H

#include <stdint.h>
#include<cublas_v2.h>
#include<mma.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cstdio>


namespace CUDA_Kernel{
    void cublasTensorOp(half *A, half *B, half *C, size_t M, size_t N, size_t K);
    void HGEMM(half*A,half*B,half*C,size_t M,size_t N,size_t K);
    void CUDA_GEMM(float*A,float*B,float*C,int M,int N,int K);
    void MV(float*A,float*x,float*y,size_t M,size_t K);
    float ReduceSum(float*A,int N);
    void SPMV(float*data,size_t*row_offset,size_t*col_index,float*x,float*y,size_t M,size_t N);
    void SPMM_88(float*data,size_t*row_offset,size_t*col_index,size_t*sort_index,float*B,float*C,size_t M,size_t N,size_t K,size_t n_rows);
    void SPMV_88(float*data,size_t*row_offset,size_t*col_index,size_t*sort_index,float*x,float*y,size_t M,size_t N,size_t n_rows);

    void SPMM_Flesible(float*data,size_t*row_offset,size_t*col_index,size_t*sort_index,float*B,float*C,size_t M,size_t N,size_t K,size_t n_rows,size_t BlockM,size_t BlockN);

    void SPMM_cuSparse(float*data,size_t*row_offset,size_t*col_index,float*B,float*C,size_t M,size_t N,size_t K,size_t n_rows);
    void HSPMM_1616(half*data,size_t*row_offset,size_t*sort_rowindex,size_t*col_index,half*B,half*C,size_t N,size_t M,size_t K,size_t n_rows);
    float SIMD_Sum(float*v,size_t n);
    void FlashAttention(float*Q,float*K,float*V,float*O,size_t N,size_t d,size_t window);
    void SGEMV(float*A,float*x,float*y,int M,int K);
    void Winograd(const float *h_img, const int N, const int C, const int H, const int W, const float *h_f, const int K, int padding = 0);
    void ReSise(unsigned char*img,unsigned char*Re_img,int H,int W,int Hout,int Wout,int C);
    void MeregeSortGPU(float*data,int n);
    void Transpose(float*M,size_t ROW,size_t COL);
    void WinogradSparseF23(float*K,float*Img,int N,int C,int H,int W,float*O,int*Kindex,int Knum);
}

#endif