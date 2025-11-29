#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda/pipeline> 
#include "Kernel.h"
namespace cg = cooperative_groups;

const int MAX_BLOCKN = 32;
// 8*8 blcok
// each warp process one block row
// each warp divide into 4 miniwarp(8 threads)
// each thread block takes 8 warps
__global__ void BSR_SPMV_88(float*data,size_t*row_offset,size_t*col_index,size_t*sort_index,float*x,float*y,size_t M,size_t N,size_t n_rows)
{
    auto block = cg::this_thread_block();
    auto miniwarp = cg::tiled_partition(block,8);
    const int lane = block.thread_rank()%8;
    const int miniwarpid = (block.thread_rank()%32)/8;

    const int rowid = cg::this_grid().num_threads()/warpSize;
    if(rowid<n_rows)
    {
        const int row_start = row_offset[rowid];
        const int row_end = row_offset[rowid+1];

        float4 subA[2];
        float4 subB[2];
        float regC = 0;
        for(int i=row_start+miniwarpid;i<row_end;i+=4)
        {
            int col_start = col_index[i];
            subA[0] = *reinterpret_cast<float4*>(&data[i*64+lane*8]);
            subB[0] = *reinterpret_cast<float4*>(&x[col_start]);
            regC += subA[0].x+subB[0].x;
            regC += subA[0].y+subB[0].y;
            regC += subA[0].z+subB[0].z;
            regC += subA[0].w+subB[0].w;

            subA[1] = *reinterpret_cast<float4*>(&data[i*64+lane*8+4]);
            subB[1] = *reinterpret_cast<float4*>(&x[col_start+4]);
            regC += subA[1].x+subB[1].x;
            regC += subA[1].y+subB[1].y;
            regC += subA[1].z+subB[1].z;
            regC += subA[1].w+subB[1].w;
        }

        //merege reduce  0-8-16-24 ->  0+8 - 8+16 - 16+24 - 24 -> 0+8+16+24 - 8+16+24 - 16+24 - 24
        regC += __shfl_down_sync(0xffffffff,regC,16);
        regC += __shfl_down_sync(0xffffffff,regC,8);
        const int row = sort_index[rowid]+lane;
        y[row] = regC;

    }
}

__global__ void BSR_SPMV_Flesible(float*data,size_t*row_offset,size_t*col_index,size_t*sort_index,float*x,float*y,size_t M,size_t N,size_t n_rows,size_t blockM,size_t blockN)
{
    //each warp porcess one blcok row and each time each warp process one
    int row_id;
    int lane;
    if(blockM<=4)
    {
        auto miniwarp = cg::tiled_partition(cg::this_thread_block(),4);
        const int miniwarpsize = 4;
        row_id =  cg::this_grid().thread_rank()/miniwarpsize;
        lane = miniwarp.thread_rank();
    }
    else if(blockM>4&&blockM<=8)
    {
        auto miniwarp = cg::tiled_partition(cg::this_thread_block(),8);
        const int miniwarpsize = 8;
        row_id =  cg::this_grid().thread_rank()/miniwarpsize;
        lane = miniwarp.thread_rank();
    }
    else if(blockM>8&&blockM<=16)
    {
        auto miniwarp = cg::tiled_partition(cg::this_thread_block(),16);
        const int miniwarpsize = 16;
        row_id =  cg::this_grid().thread_rank()/miniwarpsize;
        lane = miniwarp.thread_rank();
    }
    else
    {
        auto miniwarp = cg::this_thread_block();
        const int miniwarpsize = 32;
        row_id =  cg::this_grid().thread_rank()/miniwarpsize;
        lane = miniwarp.thread_rank();
    }


    if(row_id<n_rows&&lane<blockM)
    {
        float regC = 0;
        const int row_start = row_offset[row_id];
        const int row_end = row_offset[row_id+1];
        for(int i=row_start;i<row_end;++i)
        {
            int col_start = col_index[i];
            #pragma unroll
            for(int j=col_start;j<col_start+blockN;++j)
            {
                regC += data[i*blockM*blockN+lane*blockN+j]*x[j];
            }
        }
        const int row = sort_index[row_id]+lane;
        y[row] = regC;

    }
}


void CUDA_Kernel::SPMV_88(float*data,size_t*row_offset,size_t*col_index,size_t*sort_index,float*x,float*y,size_t M,size_t N,size_t n_rows)
{
    cudaStream_t mv;
    cudaStreamCreate(&mv);
    float*d_data,*d_x,*d_y;
    size_t*row,*col,*sort;
    const int blocksize = 8;
    cudaMallocAsync(&d_data,sizeof(float)*blocksize*blocksize*row_offset[n_rows],mv);
    cudaMallocAsync(&d_x,sizeof(float)*N,mv);
    cudaMallocAsync(&d_y,sizeof(float)*M,mv);
    cudaMallocAsync(&row,sizeof(size_t)*(n_rows+1),mv);
    cudaMallocAsync(&col,sizeof(size_t)*row_offset[n_rows],mv);
    cudaMallocAsync(&sort,sizeof(size_t)*n_rows,mv);

    dim3 Block(256);
    dim3 Grid((n_rows+7)/8);

    BSR_SPMV_88<<<Grid,Block>>>(d_data,row,col,sort,d_x,d_y,M,N,n_rows);
}