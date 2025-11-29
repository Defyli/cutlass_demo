#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include <cuda/pipeline>
#include "Kernel.h"
namespace cg = cooperative_groups;

constexpr unsigned int BLOCK_MAX_M = 16;
constexpr unsigned int BLOCK_MAX_N = 16;

template<size_t BLOCK_TileM = 256>
__global__ void SPMM_FLES(float*data,size_t*row_offset,size_t*sort_rowindex,size_t*col_index,float*B,float*C,size_t M,size_t N,size_t K,size_t n_rows,size_t BlockM,size_t BlockN)
{
    auto block = cg::this_thread_block();
    const int threadid = block.thread_rank();
    const int row_id = blockIdx.x; // one block per row
    const int offsetB = blockIdx.y*BLOCK_TileM;  // Each Block Porcess 8*128 of C
    const int warpnum = block.num_threads()/warpSize;

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, 2> shared_state;
    auto pipeline = cuda::make_pipeline(block,&shared_state);
    __shared__ float SubA[2][BLOCK_MAX_M*BLOCK_MAX_N];
    __shared__ float SubB[2][BLOCK_TileM];
    float regC[BLOCK_MAX_M] = {0};

    //No overflowing can happen since 1024*2*4 = 8KB < 100KB

    const int row_start = row_offset[row_id];
    const int row_end = row_offset[row_id+1];
    int stage_c = 0;
    if(row_end-row_start>=2)
    {
        // pipeline.producer_acquire();
        // for(int i=threadid;i<BlockM*BlockN;i+=block.num_threads())
        //     cuda::memcpy_async(block,&SubA[0][i],&data[row_start*BlockM*BlockN+i],sizeof(float),pipeline);
        // pipeline.producer_commit();

        for(int i=threadid;i<BlockM*BlockN;i+=block.num_threads())
            SubA[0][i] = data[row_start*BlockM*BlockN+i];


        for(int b=row_start+1;b<row_end;++b)
        {
            int stage = 0;
            for(int i=threadid;i<BlockM*BlockN;i+=block.num_threads())
                SubA[stage_c^1][i] = data[row_start*BlockM*BlockN+i];

            pipeline.producer_acquire();
            // for(int i=threadid;i<BlockM*BlockN;i+=block.num_threads())
            //     cuda::memcpy_async(block,&SubA[stage_c^1][i],&data[b*BlockM*BlockN+i],sizeof(float),pipeline);
            
            //fetch B
            if(threadid+offsetB<N)cuda::memcpy_async(block,&SubB[0][threadid],&B[col_index[b-1]*N+offsetB+threadid],sizeof(float),pipeline);
            pipeline.producer_commit();

            
            #pragma unroll
            for(int Bc = 1;Bc<BlockN;++Bc)
            {
                pipeline.producer_acquire();
                if(threadid+offsetB<N)cuda::memcpy_async(block,&SubB[stage^1][threadid],&B[(col_index[b-1]+Bc)*N+offsetB+threadid],sizeof(float),pipeline);
                pipeline.producer_commit();

                pipeline.consumer_wait();
                #pragma unroll
                for(int Br = 0;Br<BlockN;++Br)
                    if(threadid+offsetB<N) regC[Br] += SubA[stage_c][Br*BlockN+Bc-1]*SubB[stage][threadid];
                pipeline.consumer_release();
                stage ^=1;
            }
            pipeline.consumer_wait();
            #pragma unroll
            for(int Br = 0;Br<BlockN;++Br)
                if(threadid+offsetB<N) regC[Br] += SubA[stage_c][Br*BlockN+BlockN-1]*SubB[stage][threadid];
            pipeline.consumer_release();

            stage_c ^=1;
        }

        int stage = 0;
        pipeline.producer_acquire();
        if(threadid+offsetB<N)cuda::memcpy_async(block,&SubB[0][threadid],&B[col_index[row_end-1]*N+offsetB+threadid],sizeof(float),pipeline);
        pipeline.producer_commit();

        #pragma unroll
        for(int Bc = 1;Bc<BlockN;++Bc)
        {
            pipeline.producer_acquire();
            if(threadid+offsetB<N)cuda::memcpy_async(block,&SubB[stage^1][threadid],&B[(col_index[row_end-1]+Bc)*N+offsetB+threadid],sizeof(float),pipeline);
            pipeline.producer_commit();


            pipeline.consumer_wait();
            #pragma unroll
            for(int Br = 0;Br<BlockN;++Br)
                if(threadid+offsetB<N) regC[Br] += SubA[stage_c][Br*BlockN+Bc-1]*SubB[stage][threadid];
            pipeline.consumer_release();
            stage ^=1;
        }

        pipeline.consumer_wait();
        #pragma unroll
        for(int Br = 0;Br<BlockN;++Br)
            if(threadid+offsetB<N) regC[Br] += SubA[stage_c][Br*BlockN+BlockN-1]*SubB[stage][threadid];
        pipeline.consumer_release();

        if(threadid+offsetB<N)
            //write back
            #pragma unroll
            for(int i=0;i<BlockM;++i)
                C[(sort_rowindex[row_id]+i)*N+threadid+offsetB] = regC[i];
    }
    else if(row_end-row_start==1)
    {
        int stage = 0;
        for(int i=threadid;i<BlockM*BlockN;i+=block.num_threads())
            SubA[0][i] = data[row_start*BlockM*BlockN+i];

        pipeline.producer_acquire();
        if(threadid+offsetB<N)cuda::memcpy_async(block,&SubB[0][threadid],&B[col_index[row_start]*N+offsetB+threadid],sizeof(float),pipeline);
        pipeline.producer_commit();

        #pragma unroll
        for(int Bc = 1;Bc<BlockN;++Bc)
        {
            pipeline.producer_acquire();
            if(threadid+offsetB<N)cuda::memcpy_async(block,&SubB[stage^1][threadid],&B[(col_index[row_start]+Bc)*N+offsetB+threadid],sizeof(float),pipeline);
            pipeline.producer_commit();

            pipeline.consumer_wait();
            #pragma unroll
            for(int Br = 0;Br<BlockM;++Br)
            {
                if(threadid+offsetB<N) regC[Br] += SubA[0][Br*BlockN+Bc-1]*SubB[stage][threadid];
            }
            pipeline.consumer_release();
            stage ^=1;
        }
        pipeline.consumer_wait();
        #pragma unroll
        for(int Br = 0;Br<BlockM;++Br)
        {
            if(threadid+offsetB<N) regC[Br] += SubA[0][Br*BlockN+BlockN-1]*SubB[stage][threadid];
        }
        pipeline.consumer_release();

        if(threadid+offsetB<N)
            //write back
            #pragma unroll
            for(int i=0;i<BlockM;++i)
                C[(sort_rowindex[row_id]+i)*N+threadid+offsetB] = regC[i];
    }

}

void CUDA_Kernel::SPMM_Flesible(float*data,size_t*row_offset,size_t*col_index,size_t*sort_index,float*B,float*C,size_t M,size_t N,size_t K,size_t n_rows,size_t BlockM,size_t BlockN)
{
    cudaStream_t mm;
    cudaStreamCreate(&mm);
    float*d_data,*d_B,*d_C;
    size_t*d_row,*d_col,*d_sort;
    cudaMallocAsync(&d_data,sizeof(float)*BlockM*BlockN*row_offset[n_rows],mm);
    cudaMallocAsync(&d_row,sizeof(size_t)*(n_rows+1),mm);
    cudaMallocAsync(&d_col,sizeof(size_t)*row_offset[n_rows],mm);
    cudaMallocAsync(&d_sort,sizeof(size_t)*n_rows,mm);
    cudaMallocAsync(&d_B,sizeof(float)*K*N,mm);
    cudaMallocAsync(&d_C,sizeof(float)*M*N,mm);

    cudaMemcpyAsync(d_data,data,sizeof(float)*BlockM*BlockN*row_offset[n_rows],cudaMemcpyHostToDevice,mm);
    cudaMemcpyAsync(d_row,row_offset,sizeof(size_t)*(n_rows+1),cudaMemcpyHostToDevice,mm);
    cudaMemcpyAsync(d_col,col_index,sizeof(size_t)*row_offset[n_rows],cudaMemcpyHostToDevice,mm);
    cudaMemcpyAsync(d_sort,sort_index,sizeof(size_t)*n_rows,cudaMemcpyHostToDevice,mm);
    cudaMemcpyAsync(d_B,B,sizeof(float)*K*N,cudaMemcpyHostToDevice,mm);
    cudaMemcpyAsync(d_C,C,sizeof(float)*M*N,cudaMemcpyHostToDevice,mm);

    if(N>=512)
    {
        dim3 Block(512);
        dim3 Grid(n_rows,(N+511)/512);
        SPMM_FLES<512><<<Grid,Block,0,mm>>>(d_data,d_row,d_sort,d_col,d_B,d_C,M,N,K,n_rows,BlockM,BlockN);
    }
    else if(N>=256&&N<512)
    {
        dim3 Block(256);
        dim3 Grid(n_rows,(N+255)/256);
        SPMM_FLES<256><<<Grid,Block,0,mm>>>(d_data,d_row,d_sort,d_col,d_B,d_C,M,N,K,n_rows,BlockM,BlockN);
    }
    else
    {
        dim3 Block(128);
        dim3 Grid(n_rows,(N+127)/128);
        SPMM_FLES<128><<<Grid,Block,0,mm>>>(d_data,d_row,d_sort,d_col,d_B,d_C,M,N,K,n_rows,BlockM,BlockN);
    }

    cudaMemcpyAsync(C,d_C,sizeof(float)*N*M,cudaMemcpyDeviceToHost,mm);
    cudaFreeAsync(d_data,mm);
    cudaFreeAsync(d_row,mm);
    cudaFreeAsync(d_col,mm);
    cudaFreeAsync(d_sort,mm);
    cudaFreeAsync(d_B,mm);
    cudaFreeAsync(d_C,mm);
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