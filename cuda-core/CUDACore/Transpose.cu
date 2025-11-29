#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include "Kernel.h"
namespace cg = cooperative_groups;

template <int BLOCK_SZ>
__global__ void mat_transpose_kernel_v3(const float* idata, float* odata, int M, int N) {
    // const int bx = blockIdx.x, by = blockIdx.y;
    // const int tx = threadIdx.x, ty = threadIdx.y;

    // __shared__ float sdata[BLOCK_SZ][BLOCK_SZ+1];
    
    // int x = bx * BLOCK_SZ + tx;
    // int y = by * BLOCK_SZ + ty;

    // constexpr int ROW_STRIDE = BLOCK_SZ / NUM_PER_THREAD;

    // if (x < N) {
    //     #pragma unroll
    //     for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
    //         if (y + y_off < M) {
    //             sdata[ty + y_off][tx] = idata[(y + y_off) * N + x]; 
    //         }
    //     }
    // }
    // __syncthreads();

    // x = by * BLOCK_SZ + tx;
    // y = bx * BLOCK_SZ + ty;
    // if (x < M) {
    //     for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
    //         if (y + y_off < N) {
    //             odata[(y + y_off) * M + x] = sdata[tx][ty + y_off];
    //         }
    //     }
    // }

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ];
    
    for(int x=bx*BLOCK_SZ+tx,xwrite=by*BLOCK_SZ+tx;x<M&&xwrite<N;x+=gridDim.x*blockDim.x,xwrite+=gridDim.y*blockDim.y)
    {
        for(int y=by*BLOCK_SZ+ty,ywrite=bx*BLOCK_SZ+ty;y<N&&y<M;y+=gridDim.y*blockDim.y,ywrite+=gridDim.x*blockDim.x)
        {
            sdata[tx][ty] = idata[x*N+y];
            __syncthreads();

            odata[xwrite*M+ywrite] = sdata[ty][tx];

        }
    }

    
}

void CUDA_Kernel::Transpose(float*M,size_t ROW,size_t COL)
{
    cudaStream_t t;
    cudaStreamCreate(&t);
    float*m,*o;
    cudaMallocAsync(&m,sizeof(float)*ROW*COL,t);
    cudaMallocAsync(&o,sizeof(float)*ROW*COL,t);
    cudaMemcpyAsync(m,M,sizeof(float)*ROW*COL,cudaMemcpyHostToDevice,t);

    dim3 BLOCK(16,16);
    int gridx = (ROW+15)/16;
    int gridy = (COL+15)/16;
    int num_threads = 1;
    gridx = min(4,gridx);
    gridy = min(4,gridy);

    mat_transpose_kernel_v3<16><<<(gridx,gridy),BLOCK,0,t>>>(m,o,ROW,COL);
    cudaMemcpyAsync(M,o,sizeof(float)*ROW*COL,cudaMemcpyDeviceToHost,t);
    cudaStreamSynchronize(t);
    
    cudaError_t e = cudaGetLastError();
    if(e!=cudaSuccess)
    {
        const char *estr=cudaGetErrorString(e);
        cudaFree(m);cudaFree(o);
        printf("CUDA ERROR!\n");
        printf("%s\n",estr);
        exit(EXIT_FAILURE); 

    }
    cudaFree(m);cudaFree(o);
    
}


