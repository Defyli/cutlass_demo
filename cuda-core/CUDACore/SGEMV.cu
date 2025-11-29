#include<cooperative_groups.h>
#include<cuda_runtime.h>
#include<cstdio>
#include "Kernel.h"
namespace cg = cooperative_groups;


template<int Tile = 32>
__global__ void GEMV(float*A,float*x,float*y,int M,int K)
{
    const int lane = cg::this_thread_block().thread_rank()%Tile;

    const int row = cg::this_grid().thread_rank()/Tile;

    if(row<M)
    {
        float val=0;
        for(int i=lane;i<K;i+=Tile)
        {
            val += A[row*K+i]*x[i];
        }
        for(int i=Tile/2;i>0;i=i/2)
            val += __shfl_down_sync(0xffffffff,val,i);
        __syncwarp();
        if(lane==0)y[row] = val;
    }
}

void CUDA_Kernel::SGEMV(float*A,float*x,float*y,int M,int K)
{
    cudaStream_t mv;
    cudaStreamCreate(&mv);
    float*d_a,*d_x,*d_y;
    cudaMallocAsync(&d_a,sizeof(float)*M*K,mv);
    cudaMallocAsync(&d_x,sizeof(float)*K,mv);
    cudaMallocAsync(&d_y,sizeof(float)*M,mv);

    cudaMemcpyAsync(d_a,A,sizeof(float)*M*K,cudaMemcpyHostToDevice,mv);
    cudaMemcpyAsync(d_x,x,sizeof(float)*K,cudaMemcpyHostToDevice,mv);
    cudaMemcpyAsync(d_y,y,sizeof(float)*M,cudaMemcpyHostToDevice,mv);


    dim3 block(256);

    if(K>32)
    {
        dim3 grid(M/8);
        GEMV<32><<<grid,block,0,mv>>>(d_a,d_x,d_y,M,K);
    }
    else if(K>=16)
    {
        dim3 grid(M/16);
        GEMV<16><<<grid,block,0,mv>>>(d_a,d_x,d_y,M,K);
    }
    else if(K>=8)
    {
        dim3 grid(M/32);
        GEMV<8><<<grid,block,0,mv>>>(d_a,d_x,d_y,M,K);
    }
    else if(K>=4)
    {
        dim3 grid(M/64);
        GEMV<4><<<grid,block,0,mv>>>(d_a,d_x,d_y,M,K);
    }

    cudaMemcpyAsync(y,d_y,sizeof(float)*M,cudaMemcpyDeviceToHost,mv);

    cudaStreamSynchronize(mv);

    cudaError_t e = cudaGetLastError();
    if(e!=cudaSuccess)
    {
        const char *estr=cudaGetErrorString(e);
        printf("CUDA ERROR!\n");
        printf("%s\n",estr);
        cudaFree(d_a);cudaFree(d_x);cudaFree(d_y);
        exit(EXIT_FAILURE);
    }
    cudaFree(d_a);cudaFree(d_x);cudaFree(d_y);
}
