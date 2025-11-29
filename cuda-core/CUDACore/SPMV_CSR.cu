#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include <cub/cub.cuh> 
#include "Kernel.h"
namespace cg = cooperative_groups;


template <class T>
__device__ T warp_reduce (T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T>
__global__ void CSR_SPMV_THREAD(size_t n_rows,size_t*row_offset,size_t*col_index,T*data,T*x,T*y)
{
    auto grid = cg::this_grid();
    const int row_id = grid.thread_rank();
    if(row_id<n_rows)
    {
        const int row_start = row_offset[row_id];
        const int row_end = row_offset[row_id+1];

        T sum = 0;
        for(int i=row_start;i<row_end;++i)
            sum += data[i]*y[col_index[i]];
        
        y[row_id] = sum;
    }

}

template<typename T>
__global__ void CSR_SPMV_WARP(size_t n_rows,size_t*row_offset,size_t*col_index,T*data,T*x,T*y)
{
    auto block = cg::this_thread_block();
    const int warpid = block.thread_rank()/warpSize;
    const int lane = block.thread_rank()%warpSize;

    T sum = 0;

    if(warpid<n_rows)
    {
        const int row_start = row_offset[warpid];
        const int row_end = row_offset[warpid+1];
        for(size_t i = row_start+lane;i<row_end;i+=warpSize)
            sum += data[i]*x[col_index[i]];
        
        sum = warp_reduce<T>(sum);

        if(lane%warpSize==0)
            y[warpid] = sum;
    }

}


// template <unsigned int WarpSize,typename T>
// __device__ __forceinline__ T warpReduceSum(T sum) {
//     if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
//     if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
//     if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
//     if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
//     if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
//     return sum;
// }


// template<UI THREADS_PER_VECTOR = 4,UI VECTOR_PER_BLOCK = 16,UI Aligened_Load_Size = 8,typename T>
// __global__ void CSR_SPMV_VECTOR(size_t n_rows,size_t*row_offset,size_t*col_index,T*data,T*x,T*y)
// {
//     //Aligened_Load_Size should be ensured that each row can be coverd. It is also recommanded to keep aligened
//     auto block = cg::this_thread_block();
//     const int warpid = block.thread_rank()/warpSize;
//     const int lane = block.thread_rank()%warpSize;
//     const int row_id = block.thread_rank()/THREADS_PER_VECTOR;

//     using WarpLoadT = WarpLoad<T,Aligened_Load_Size,cub::WARP_LOAD_VECTORIZE,THREADS_PER_VECTOR>;

//     if(row_id<n_rows)
//     {
//         const int row_start = row_offset[row_id];
//         const int row_end = row_offset[row_id+1];

//         __shared__ typename WarpLoadT::TempStorage temp_storage[VECTOR_PER_BLOCK/THREADS_PER_VECTOR];

//         T localdata[Aligened_Load_Size];
//         T localindex[Aligened_Load_Size];
//         WarpLoadT(temp_storage[row_id]).Load(data+row_start,localdata);
//         __syncwarp();
//         WarpLoadT(temp_storge[row_id]).Load(col_index+row_start,localindex);
//         __syncwarp();

//         T sum = 0;

//         for(int i = lane%THREADS_PER_VECTOR;i<row_end-row_start;i+=THREADS_PER_VECTOR)
//         {
//             sum += localdata[i]*y[localindex[i]];   
//         }
//         sum = warpReduceSum<THREADS_PER_VECTOR,T>(sum);
//         if(lane%THREADS_PER_VECTOR ==0)
//             y[row_id] = sum;
//     }
// }



void CUDA_Kernel::SPMV(float*data,size_t*row_offset,size_t*col_index,float*x,float*y,size_t M,size_t N)
{
    float*d_data,*d_x,*d_y;
    size_t*d_row,*d_col;
    cudaSetDevice(0);
    cudaStream_t mv;
    cudaStreamCreate(&mv);
    cudaMallocAsync(&d_data,sizeof(float)*row_offset[N],mv);
    cudaMallocAsync(&d_row,sizeof(size_t)*(N+1),mv);
    cudaMallocAsync(&d_col,sizeof(size_t)*row_offset[N],mv);
    cudaMallocAsync(&d_x,sizeof(float)*N,mv);
    cudaMallocAsync(&d_y,sizeof(float)*N,mv);

    cudaMemcpyAsync(d_data,data,sizeof(float)*row_offset[N],cudaMemcpyDeviceToHost,mv);
    cudaMemcpyAsync(d_row,row_offset,sizeof(size_t)*(N+1),cudaMemcpyHostToDevice,mv);
    cudaMemcpyAsync(d_col,col_index,sizeof(size_t)*row_offset[N],cudaMemcpyHostToDevice,mv);
    cudaMemcpyAsync(d_x,x,sizeof(float)*N,cudaMemcpyDeviceToHost,mv);
    cudaMemcpyAsync(d_y,y,sizeof(float)*N,cudaMemcpyHostToDevice,mv);

    if(N<=128)
    {
        dim3 block(256);
        dim3 grid((N+255)/256);

        CSR_SPMV_THREAD<float><<<grid,block>>>(N,d_row,d_col,d_data,x,y);
    }
    else
    {

        dim3 block(256);
        dim3 grid((N+7)/256);
        CSR_SPMV_THREAD<float><<<grid,block>>>(N,d_row,d_col,d_data,x,y);

    }

    cudaMemcpyAsync(y,d_y,sizeof(float)*M,cudaMemcpyDeviceToHost,mv);
    cudaFreeAsync(d_data,mv);
    cudaFreeAsync(d_x,mv);
    cudaFreeAsync(d_y,mv);
    cudaFreeAsync(d_row,mv);
    cudaFreeAsync(d_col,mv);
    cudaStreamSynchronize(mv);

    cudaError_t e = cudaGetLastError();
    if(e!=cudaSuccess)
    {
        const char *estr=cudaGetErrorString(e);
        printf("CUDA ERROR!\n");
        printf("%s\n",estr);
        exit(EXIT_FAILURE); 

    }

}