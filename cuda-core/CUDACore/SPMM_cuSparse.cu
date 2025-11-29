#include<cusparse.h>
#include<cusparse_v2.h>
#include<cuda_runtime.h>
#include<iostream>
#include "Kernel.h"
void CUDA_Kernel::SPMM_cuSparse(float*data,size_t*row_offset,size_t*col_index,float*B,float*C,size_t M,size_t N,size_t K,size_t n_rows)
{

    int*d_row,*d_col;
    float*d_val,*d_b,*d_c;
    int bnnz = row_offset[n_rows];
    const int blocksize = 8;
    cudaMalloc(&d_row,sizeof(int)*(n_rows+1));
    cudaMalloc(&d_col,sizeof(int)*(bnnz));
    cudaMalloc(&d_val,sizeof(float)*bnnz*blocksize*blocksize);
    cudaMalloc(&d_b,sizeof(float)*K*M);
    cudaMalloc(&d_c,sizeof(float)*M*N);
    cudaMemcpy(d_row,row_offset,sizeof(int)*(n_rows+1),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col,col_index,sizeof(int)*bnnz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_val,data,sizeof(float)*bnnz*blocksize*blocksize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,B,sizeof(float)*K*M,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,C,sizeof(float)*M*N,cudaMemcpyHostToDevice);

    
    cusparseHandle_t h;
    cusparseCreate(&h);
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    float alpha=1;
    float bete = 0;
    // cusparseSpMatDescr_t mat1;
    // auto m1=cusparseCreateBsr(&mat1,n_rows,N/blocksize,bnnz,blocksize,blocksize,d_row,d_col,d_val,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    cusparseStatus_t res=cusparseSbsrmm(h,CUSPARSE_DIRECTION_ROW,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,n_rows,M,M/blocksize,bnnz,&alpha,descr,d_val,d_row,d_col,blocksize,d_b,M,&bete,d_c,M);
    if(res!=CUSPARSE_STATUS_SUCCESS)
    {
        
        std::cout<<"Cusparse Fail! "<<cusparseGetErrorName(res)<<std::endl;
        cudaFree(d_row);
        cudaFree(d_col);
        cudaFree(d_val);
        cudaFree(d_b);
        cudaFree(d_c);
        exit(-1);
    }

    cudaMemcpy(C,d_c,sizeof(float)*M*N,cudaMemcpyDeviceToDevice);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_val);
    cudaFree(d_b);
    cudaFree(d_c);

}