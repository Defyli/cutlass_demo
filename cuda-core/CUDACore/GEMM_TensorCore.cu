#include<cuda_runtime.h>
#include<mma.h>
#include<cstdlib>
#include<cuda.h>
#include<cublas_v2.h>
#include<stdio.h>
#include "Kernel.h"
using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_SIZE = 32;
constexpr int Tile = 128;  // compute 128*128
constexpr int THREADS_PER_BLOCK = 256; // 8 Warps
constexpr int WARP_PER_BLOCK = 8;

// Warp Tile 
constexpr int WARP_TILE_ROW = 4;
constexpr int WARP_TILE_COL = 2;

// Work Group Tile
constexpr int Work_M = 2; // 8/4
constexpr int Work_N = 4; // 8/2
typedef float4 copy_t;

__global__ void TensorCoreGemm(const half*A,const half*B,half*C,size_t M,size_t N,size_t K)
{
   int Warpid = threadIdx.x/WARP_SIZE;
   int Threadid = threadIdx.x%32;
   int BaseX = blockIdx.x*WARP_PER_BLOCK*WMMA_M;
   int BaseY = blockIdx.y*WARP_PER_BLOCK*WMMA_N;

   __shared__ half subA[Work_M*WARP_TILE_ROW][WMMA_M*WMMA_K];// 8*(16*16)
   __shared__ half subB[Work_N*WARP_TILE_COL][WMMA_N*WMMA_K];// 8*(16*16)

   wmma::fragment<wmma::accumulator,WMMA_M,WMMA_N,WMMA_K,half>c_flag[Work_M][Work_N];
#pragma unroll
    for(int i=0;i<Work_M;++i)
    {
        for(int j=0;j<Work_N;++j)
            wmma::fill_fragment(c_flag[i][j],0.0);
    }

#pragma unroll
   for(int i = 0;i<K;i+=WMMA_K)
   {
        //move data from global to cache
        
        int RowA = BaseY+Warpid*WMMA_M+Threadid/2;
        int ColA = i+(Threadid%2)*8;
        int RowB = i+Threadid/2;
        int ColB = BaseX+Warpid*WMMA_N+(Threadid%2)*8;
        //A
        *reinterpret_cast<copy_t*>(&subA[Warpid][Threadid*8]) = *reinterpret_cast<const copy_t*>(&A[RowA*K+ColA]);

        //B
        *reinterpret_cast<copy_t*>(&subB[Warpid][Threadid*8]) = *reinterpret_cast<const copy_t*>(&B[RowB*N+ColB]);

        __syncthreads();
        for(int ii = 0;ii<Work_M;++ii)
        {
            wmma::fragment<wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major>a_flag;
            wmma::load_matrix_sync(a_flag,&subA[int(Warpid/2)*2+ii][0],WMMA_K);

            for(int jj = 0;jj<Work_N;++jj)
            {
                wmma::fragment<wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major>b_flag;
                wmma::load_matrix_sync(b_flag,&subB[(Warpid%2)*4+jj][0],WMMA_N);

                wmma::mma_sync(c_flag[ii][jj],a_flag,b_flag,c_flag[ii][jj]);
            }
        }
   }

#pragma unroll
    for(int i=0;i<Work_M;++i)
    {
        for(int j=0;j<Work_N;++j)
        {
            int Row = BaseY+(int(Warpid/2)*2+i)*WMMA_M;
            int Col = BaseX+((Warpid%2)*4+j)*WMMA_N;
            wmma::store_matrix_sync(C+Row*N+Col,c_flag[i][j],N,wmma::mem_row_major);
        }
    }

}

void CUDA_Kernel::HGEMM(half*A,half*B,half*C,size_t M,size_t N,size_t K)
{
    cudaSetDevice(0);

    half*d_a,*d_b,*d_c;
    cudaMallocAsync(&d_a,sizeof(half)*M*K,0);
    cudaMallocAsync(&d_b,sizeof(half)*N*K,0);
    cudaMallocAsync(&d_c,sizeof(half)*N*M,0);

    cudaMemcpyAsync(d_a,A,sizeof(half)*M*K,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_b,B,sizeof(half)*N*K,cudaMemcpyHostToDevice,0);

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((M+Tile-1)/Tile,(N+Tile-1)/Tile);

    TensorCoreGemm<<<grid,block>>>(d_a,d_b,d_c,M,N,K);
    cudaMemcpyAsync(C,d_c,sizeof(half)*M*N,cudaMemcpyDeviceToHost,0);
    cudaFreeAsync(d_a,0);
    cudaFreeAsync(d_b,0);
    cudaFreeAsync(d_c,0);

    cudaStreamSynchronize(0);
    
    cudaError_t e = cudaGetLastError();
    if(e!=cudaSuccess)
    {
        const char *estr=cudaGetErrorString(e);
        printf("CUDA ERROR!\n");
        printf("%s\n",estr);
        exit(EXIT_FAILURE);
    }
}


