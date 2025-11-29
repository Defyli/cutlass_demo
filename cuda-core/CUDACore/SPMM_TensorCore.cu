#include<cuda_runtime.h>
#include<mma.h>
#include<cstdlib>
#include<cuda.h>
#include<cublas_v2.h>
#include<stdio.h>
#include "Kernel.h"
#include<cooperative_groups.h>
#include <cuda/pipeline>
namespace wmma = nvcuda::wmma;
namespace cg = cooperative_groups;
constexpr int WMMAN = 16;
constexpr int WMMAM = 16;
constexpr int WMMAK = 16;
constexpr int BLOCK_TileM = 128;
//each block process 16*K 

__global__ void SPMM_TC(half*data,size_t*row_offset,size_t*sort_rowindex,size_t*col_index,half*B,half*C,size_t N,size_t M,size_t K,size_t n_rows)
{

    auto block = cg::this_thread_block();
    const int threadid = block.thread_rank();
    const int warpid = threadid/warpSize;
    const int row_id = blockIdx.x;
    const int offsetB = blockIdx.y*BLOCK_TileM;

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block,2>shared_state;
    auto pipline = cuda::make_pipeline(block,&shared_state);

    const size_t row_start = row_offset[row_id];
    const size_t row_end = row_offset[row_id+1];

    if(row_id<n_rows&&row_start<row_end)
    {
        int stage = 0;
        size_t col_start = col_index[row_start*WMMAK];
        wmma::fragment<wmma::accumulator,WMMAM,WMMAN,WMMAK,half>C_flag;
        if(row_end-row_start>=2)
        {
            wmma::fragment<wmma::matrix_a,WMMAM,WMMAN,WMMAK,half,wmma::row_major>A_flag[2];
            wmma::fragment<wmma::matrix_b,WMMAM,WMMAM,WMMAK,half,wmma::row_major>B_flag[2];

            wmma::fill_fragment(C_flag,0.0);

            wmma::load_matrix_sync(A_flag[stage],&data[row_start*WMMAM*WMMAK],WMMAK);
            wmma::load_matrix_sync(B_flag[stage],B+col_start*M+warpid*WMMAN+offsetB,M);

            for(size_t i=row_start+1;i<row_end;++i)
            {
                wmma::mma_sync(C_flag,A_flag[stage],B_flag[stage],C_flag);
                col_start = col_index[i*WMMAK];
                wmma::load_matrix_sync(A_flag[stage^1],&data[i*WMMAK*WMMAK],WMMAK);
                wmma::load_matrix_sync(B_flag[stage^1],B+col_start*M+offsetB+warpid*WMMAN,M);
                stage ^=1;
            }
            wmma::mma_sync(C_flag,A_flag[stage],B_flag[stage],C_flag);
        }
        else
        {
            wmma::fragment<wmma::matrix_a,WMMAM,WMMAN,WMMAK,half,wmma::row_major>A_flag;
            wmma::fragment<wmma::matrix_b,WMMAM,WMMAN,WMMAK,half,wmma::row_major>B_flag;

            wmma::load_matrix_sync(A_flag,&data[row_start*WMMAM*WMMAK],WMMAK);
            wmma::load_matrix_sync(B_flag,B+col_start*M+warpid*WMMAN+offsetB,M);
            wmma::mma_sync(C_flag,A_flag,B_flag,C_flag);
        }
        size_t row = sort_rowindex[row_id];
        size_t col = warpid*WMMAM+offsetB;
        wmma::store_matrix_sync(C+row*M+col,C_flag,M,wmma::mem_row_major);
    }

}

void CUDA_Kernel::HSPMM_1616(half*data,size_t*row_offset,size_t*sort_index,size_t*col_index,half*B,half*C,size_t N,size_t M,size_t K,size_t n_rows)
{
    half*d_a,*d_b,*d_c;
    size_t*d_row,*d_col,*d_sort;
    const int blocksize = 16;

    cudaSetDevice(0);
    cudaStream_t mm;
    cudaStreamCreate(&mm);
    cudaMallocAsync(&d_a,sizeof(half)*row_offset[n_rows]*blocksize*blocksize,mm);
    cudaMallocAsync(&d_b,sizeof(half)*K*M,mm);
    cudaMallocAsync(&d_c,sizeof(half)*M*N,mm);
    cudaMallocAsync(&d_row,sizeof(size_t)*(n_rows+1),mm);
    cudaMallocAsync(&d_col,sizeof(size_t)*blocksize*row_offset[n_rows],mm);
    cudaMallocAsync(&d_sort,sizeof(size_t)*n_rows,mm);

    cudaMemcpyAsync(d_a,data,sizeof(half)*row_offset[n_rows]*blocksize*blocksize,cudaMemcpyHostToDevice,mm);
    cudaMemcpyAsync(d_b,B,sizeof(half)*K*M,cudaMemcpyHostToDevice,mm);
    cudaMemcpyAsync(d_c,C,sizeof(half)*M*N,cudaMemcpyHostToDevice,mm);
    cudaMemcpyAsync(d_row,row_offset,sizeof(size_t)*(n_rows+1),cudaMemcpyHostToDevice,mm);
    cudaMemcpyAsync(d_col,col_index,sizeof(size_t)*blocksize*row_offset[n_rows],cudaMemcpyHostToDevice,mm);
    cudaMemcpyAsync(d_sort,sort_index,sizeof(size_t)*n_rows,cudaMemcpyHostToDevice,mm);

    dim3 Block(256,1);
    dim3 grid(n_rows,(M+BLOCK_TileM-1)/BLOCK_TileM);
    SPMM_TC<<<grid,Block>>>(d_a,d_row,d_sort,d_col,d_b,d_c,N,M,K,n_rows);

    cudaMemcpyAsync(C,d_c,sizeof(half)*N*M,cudaMemcpyDeviceToHost,mm);
    cudaFreeAsync(d_a,mm);
    cudaFreeAsync(d_b,mm);
    cudaFreeAsync(d_c,mm);
    cudaFreeAsync(d_row,mm);
    cudaFreeAsync(d_col,mm);
    cudaFreeAsync(d_sort,mm);
    cudaStreamSynchronize(mm);

    cudaError_t e = cudaGetLastError();
    if(e!=cudaSuccess)
    {
        const char *estr=cudaGetErrorString(e);
        printf("CUDA ERROR!\n");
        printf("%s\n",estr);
        exit(EXIT_FAILURE); 

    }
}