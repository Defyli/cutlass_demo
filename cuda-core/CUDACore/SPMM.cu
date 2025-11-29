#include<stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include<cub/cub.cuh>
#include <cub/warp/warp_load.cuh>
#include "Kernel.h"
namespace cg = cooperative_groups;
// 为了保证SM内负载较为均衡，需要对稀疏矩阵进行sort，按每行的block数排序，从大到小。
// data 变为二维数组，第二维遵循对其原则,必须padding到8*8或者16*16;
// 由于block方阵属性col_index只记录第一行的列坐标，且也需要pading到8/16;
// sort_index 记录了按照行中块数量排序（大->小）后的行号

constexpr int Per_Warp8 = 8; //each warp process 8*8 blocks as default (合并到8或者切分到8)

constexpr int Warp_Per_Block = 8; //each block contains 256 threads
constexpr int BLOCK_TileM = 128; // 32*8
constexpr int Warp_TilN = 1;
constexpr int Thread_M = 32;
constexpr int Thread_N = 1;  // Each Warp Process 
// 8*8 Version
__global__ void SPMM_BLOCK_8(float*data,size_t*row_offset,size_t*sort_rowindex,size_t*col_index,float*B,float*C,
                             size_t N,size_t M,size_t K,size_t n_rows)
{
    auto block = cg::this_thread_block();
    const int threadid = block.thread_rank();
    const int lane = threadid%warpSize;
    const int warpid = threadid/warpSize;
    const int row_id = blockIdx.x; // one block per row
    const int offsetB = blockIdx.y*BLOCK_TileM;  // Each Block Porcess 8*128 of C
    // double buffer for B (No Need)
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, 2> shared_state;
    auto pipeline = cuda::make_pipeline(block,&shared_state);

    __shared__ float BlockA[2][Per_Warp8*Per_Warp8];

    const int row_start = row_offset[row_id];
    const int row_end = row_offset[row_id+1];


    if(row_id<n_rows&&row_start<row_end)
    {
        int stage = 0;
        float4 regB[2]; //1
        float4 regC = {0,0,0,0};
        int col_start = col_index[row_start];// in fact, we just need the first val and we could infer the res 7 elements
        if(row_end-row_start>=2)
        {
            pipeline.producer_acquire();
            cuda::memcpy_async(block,&regB[stage],reinterpret_cast<float4*>(B+(col_start+warpid)*M+offsetB+lane*4),sizeof(float4),pipeline);
            if(warpid==0) cuda::memcpy_async(block,reinterpret_cast<float2*>(&BlockA[stage][lane*2]),reinterpret_cast<float2*>(&data[row_start*Per_Warp8*Per_Warp8+lane*2]),sizeof(float2),pipeline);
            pipeline.producer_commit();

            for(size_t b = row_start+1;b<row_end;++b)
            {

                pipeline.consumer_wait();
                for(int i=0;i<Per_Warp8;++i)
                {
                    float regA = BlockA[stage][warpid*8+i];
                    regC.x += regA*regB[stage].x;
                    regC.y += regA*regB[stage].y;
                    regC.z += regA*regB[stage].z;
                    regC.w += regA*regB[stage].w;
                }
                pipeline.consumer_release();

                col_start = col_index[b];
                stage ^=1;
                pipeline.producer_acquire();
                cuda::memcpy_async(block,&regB[stage],reinterpret_cast<float4*>(B+(col_start+warpid)*M+offsetB+lane*4),sizeof(float4),pipeline);
                if(warpid==0) cuda::memcpy_async(block,reinterpret_cast<float2*>(&BlockA[stage][lane*2]),reinterpret_cast<float2*>(&data[b*Per_Warp8*Per_Warp8+lane*2]),sizeof(float2),pipeline);
                pipeline.producer_commit();

            }

            pipeline.consumer_wait();
            for(int i=0;i<Per_Warp8;++i)
            {       
                float regA = BlockA[stage][warpid*8+i];
                regC.x += regA*regB[stage].x;
                regC.y += regA*regB[stage].y;
                regC.z += regA*regB[stage].z;
                regC.w += regA*regB[stage].w;

            }
            pipeline.consumer_release();

        }
        else
        {
            //copy A
            if(warpid==0) *reinterpret_cast<float2*>(BlockA[0]) = *reinterpret_cast<float2*>(&data[row_start*Per_Warp8*Per_Warp8]);
            regB[0] = *reinterpret_cast<float4*>(B+(col_start+warpid)*M+offsetB+lane*4);

            for(int i=0;i<Per_Warp8;++i)
            {
                float regA = BlockA[0][warpid*8+i];
                regC.x += regA*regB[0].x;
                regC.y += regA*regB[0].y;
                regC.z += regA*regB[0].z;
                regC.w += regA*regB[0].w;
            } 
        }

        //write C
        int row = sort_rowindex[row_id]+warpid;
        int col = offsetB+lane*4;
        *reinterpret_cast<float4*>(C+row*M+col) = regC;
    }

}


void CUDA_Kernel::SPMM_88(float*data,size_t*row_offset,size_t*col_index,size_t*sort_index,float*B,float*C,size_t M,size_t N,size_t K,size_t n_rows)
{
    float*d_a,*d_b,*d_c;
    size_t*d_row,*d_col,*d_sort;
    const int blocksize = 8;

    cudaSetDevice(0);
    cudaStream_t mm;
    cudaStreamCreate(&mm);
    cudaMallocAsync(&d_a,sizeof(float)*row_offset[n_rows]*blocksize*blocksize,mm);
    cudaMallocAsync(&d_b,sizeof(float)*K*M,mm);
    cudaMallocAsync(&d_c,sizeof(float)*M*N,mm);
    cudaMallocAsync(&d_row,sizeof(size_t)*(n_rows+1),mm);
    cudaMallocAsync(&d_col,sizeof(size_t)*row_offset[n_rows],mm);
    cudaMallocAsync(&d_sort,sizeof(size_t)*n_rows,mm);

    cudaMemcpyAsync(d_a,data,sizeof(float)*row_offset[n_rows]*blocksize*blocksize,cudaMemcpyHostToDevice,mm);
    cudaMemcpyAsync(d_b,B,sizeof(float)*K*M,cudaMemcpyHostToDevice,mm);
    cudaMemcpyAsync(d_c,C,sizeof(float)*M*N,cudaMemcpyHostToDevice,mm);
    cudaMemcpyAsync(d_row,row_offset,sizeof(size_t)*(n_rows+1),cudaMemcpyHostToDevice,mm);
    cudaMemcpyAsync(d_col,col_index,sizeof(size_t)*row_offset[n_rows],cudaMemcpyHostToDevice,mm);
    cudaMemcpyAsync(d_sort,sort_index,sizeof(size_t)*n_rows,cudaMemcpyHostToDevice,mm);

    dim3 Block(256,1);
    dim3 grid(n_rows,(M+BLOCK_TileM-1)/BLOCK_TileM);
    SPMM_BLOCK_8<<<grid,Block>>>(d_a,d_row,d_sort,d_col,d_b,d_c,N,M,K,n_rows);

    cudaMemcpyAsync(C,d_c,sizeof(float)*N*M,cudaMemcpyDeviceToHost,mm);
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
