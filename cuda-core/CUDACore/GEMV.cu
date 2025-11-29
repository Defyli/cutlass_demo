#include<stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include "Kernel.h"

constexpr int ThreadY = 16;
constexpr int ThreadX = 32;

template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}


__global__ void GEMV(float*A,float*x,float*y,size_t M,size_t K)
{
    int bx = blockIdx.x;
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int warp_size=32;
    int laneId= tx % warp_size;
    int current_row = blockDim.y * bx + ty;

    if(current_row < M){
        float res=0;
        int kIteration = K/warp_size;
        if(kIteration==0) kIteration=1;
        #pragma unroll
        for(int i=0; i< kIteration; i++){
            int current_col = i*warp_size + laneId;
            res += A[current_row*K + current_col] * x[current_col];
        }
        res = warpReduceSum<warp_size>(res);
        if(laneId==0) y[current_row]=res;
    }
}

void CUDA_Kernel::MV(float*A,float*x,float*y,size_t M,size_t K)
{
    float *d_a,*d_x,*d_y;
    cudaMallocAsync(&d_a,sizeof(float)*M*K,0);
    cudaMallocAsync(&d_x,sizeof(float)*K,0);
    cudaMallocAsync(&d_y,sizeof(float)*M,0);

    cudaMemcpyAsync(d_a,A,sizeof(float)*M*K,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_x,x,sizeof(float)*K,cudaMemcpyHostToDevice,0);

    dim3 grid((M+ThreadY-1)/ThreadY,1);
    dim3 block(ThreadX,ThreadY);

    GEMV<<<grid,block>>>(d_a,d_x,d_y,M,K);

    cudaMemcpyAsync(y,d_y,sizeof(float)*M,cudaMemcpyDeviceToHost,0);
    cudaFreeAsync(d_a,0);
    cudaFreeAsync(d_x,0);
    cudaFreeAsync(d_y,0);
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