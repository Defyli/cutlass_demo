#include<stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include "Kernel.h"
const size_t BLOCK_M = 128;
const size_t BLOCK_N = 128;
const size_t BLOCK_SIZE = 16;
const size_t BLOCK_K = 8;
const size_t Per_M = BLOCK_M/BLOCK_SIZE;
const size_t Per_N = BLOCK_N/BLOCK_SIZE;

void __global__ GEMM(float *A,float *B,float*C,size_t M,size_t N,size_t K)
{
    
    const size_t CX = blockIdx.x * blockDim.x * Per_N;
    const size_t CY = blockIdx.y * blockDim.y * Per_M;
    const size_t baseIdx = threadIdx.y*blockDim.x+threadIdx.x;
    

    auto block = cooperative_groups::this_thread_block();
    constexpr auto scope = cuda::thread_scope_block;
    constexpr auto stages_count = 2;
    __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
    __shared__ float subA[2][BLOCK_M*BLOCK_K];
    __shared__ float subB[2][BLOCK_K*BLOCK_N];

    //for read from the Global Mem
    float4 regA[2][Per_M/4];
    float4 regB[2][Per_N/4];

    float *baseC = C+ (CY+threadIdx.y*Per_M)*N + CX+ threadIdx.x*Per_N;
    float *baseA = A + CY*K;
    float *baseB = B + CX;

    int rowA = baseIdx/2;
    int colA = (baseIdx%(BLOCK_K/4))*4;
    int rowB = baseIdx/(BLOCK_N/4);
    int colB = (baseIdx*4)%BLOCK_N;

    float c[Per_M*Per_N] = {};


    auto pipeline = cuda::make_pipeline(block,&shared_state);
    // double buffer

    cuda::memcpy_async(subB[0]+baseIdx * 4,baseB+rowB*N+colB,sizeof(float4),pipeline);

    cuda::memcpy_async(subA[0]+rowA + colA * BLOCK_M,baseA+rowA*K+colA, sizeof(float), pipeline);
    cuda::memcpy_async(subA[0]+rowA + (colA + 1) * BLOCK_M,baseA+rowA*K+colA+1, sizeof(float), pipeline);
    cuda::memcpy_async(subA[0] + rowA + (colA + 2) * BLOCK_M, baseA+rowA*K+colA+2, sizeof(float), pipeline);
    cuda::memcpy_async(subA[0] + rowA + (colA + 3) * BLOCK_M, baseA+rowA*K+colA+3, sizeof(float), pipeline);

    int memstage = 0;
    pipeline.producer_commit();


    for(int i = BLOCK_K;i < K; i+=BLOCK_K)
    {

        //load mem from the global mem and store them
        pipeline.producer_acquire();

        cuda::memcpy_async(subB[memstage^1] + baseIdx * 4, baseB + i * N + colB * N + rowB, sizeof(float4), pipeline);

        cuda::memcpy_async(&subA[memstage^1][rowA + colA * BLOCK_M], baseA + i + colA * K + rowA, sizeof(float), pipeline);
        cuda::memcpy_async(&subA[memstage^1][rowA + (colA + 1) * BLOCK_M], baseA + i + colA * K + rowA+1, sizeof(float), pipeline);
        cuda::memcpy_async(&subA[memstage^1][rowA + (colA + 2) * BLOCK_M], baseA + i + colA * K + rowA+2, sizeof(float), pipeline);
        cuda::memcpy_async(&subA[memstage^1][rowA + (colA + 3) * BLOCK_M], baseA + i + colA * K + rowA+3, sizeof(float), pipeline);
        pipeline.producer_commit();

        pipeline.consumer_wait();

        int stage = 0;
#pragma unroll
        for(int ii=0;ii<BLOCK_K; ii++)
        {

            regA[stage][0] = *reinterpret_cast<float4 *>(&subA[memstage][(threadIdx.x*Per_M)+ ii*BLOCK_M]);
            regA[stage][1] = *reinterpret_cast<float4 *>(&subA[memstage][(threadIdx.x*Per_M+4)+ii*BLOCK_M]);

            regB[stage][0] = *reinterpret_cast<float4 *>(&subB[memstage][threadIdx.y * Per_N+BLOCK_N*ii]);
            regB[stage][1] = *reinterpret_cast<float4 *>(&subB[memstage][threadIdx.y * Per_N +4+ BLOCK_N*ii]);

#pragma unroll
            for(int cpi = 0;cpi < Per_M/4;cpi++)
            {
                for(int cpj = 0;cpj < Per_N/4;cpj++)
                {
                    c[cpi * 4 * Per_M + cpj * 4] += regA[stage][cpi].x * regB[stage][cpj].x;
                    c[cpi * 4 * Per_M + cpj * 4 + 1] += regA[stage][cpi].x * regB[stage][cpj].y;
                    c[cpi * 4 * Per_M + cpj * 4 + 2] += regA[stage][cpi].x * regB[stage][cpj].z;
                    c[cpi * 4 * Per_M + cpj * 4 + 3] += regA[stage][cpi].x * regB[stage][cpj].w;

                    c[(cpi * 4 + 1) * Per_M + cpj * 4] += regA[stage][cpi].y * regB[stage][cpj].x;
                    c[(cpi * 4 + 1) * Per_M + cpj * 4 + 1] += regA[stage][cpi].y * regB[stage][cpj].y;
                    c[(cpi * 4 + 1) * Per_M + cpj * 4 + 2] += regA[stage][cpi].y * regB[stage][cpj].z;
                    c[(cpi * 4 + 1) * Per_M + cpj * 4 + 3] += regA[stage][cpi].y * regB[stage][cpj].w;

                    c[(cpi * 4 + 2) * Per_M + cpj * 4] += regA[stage][cpi].z * regB[stage][cpj].x;
                    c[(cpi * 4 + 2) * Per_M + cpj * 4 + 1] += regA[stage][cpi].z * regB[stage][cpj].y;
                    c[(cpi * 4 + 2) * Per_M + cpj * 4 + 2] += regA[stage][cpi].z * regB[stage][cpj].z;
                    c[(cpi * 4 + 2) * Per_M + cpj * 4 + 3] += regA[stage][cpi].z * regB[stage][cpj].w;

                    c[(cpi * 4 + 3) * Per_M + cpj * 4] += regA[stage][cpi].w * regB[stage][cpj].x;
                    c[(cpi * 4 + 3) * Per_M + cpj * 4 + 1] += regA[stage][cpi].w * regB[stage][cpj].y;
                    c[(cpi * 4 + 3) * Per_M + cpj * 4 + 2] += regA[stage][cpi].w * regB[stage][cpj].z;
                    c[(cpi * 4 + 3) * Per_M + cpj * 4 + 3] += regA[stage][cpi].w * regB[stage][cpj].w;
                }
            }
            stage ^=1;
        }
        pipeline.consumer_release();
        memstage ^=1;
    }

    // last compute
    pipeline.consumer_wait();
 
    int stage = 0;
#pragma unroll
    for(int ii=0;ii<BLOCK_K; ii++)
    {

        regA[stage][0] = *reinterpret_cast<float4 *>(&subA[memstage][(threadIdx.x*Per_M)+ ii*BLOCK_M]);
        regA[stage][1] = *reinterpret_cast<float4 *>(&subA[memstage][(threadIdx.x*Per_M+4)+ii*BLOCK_M]);

        regB[stage][0] = *reinterpret_cast<float4 *>(&subB[memstage][threadIdx.y * Per_N+BLOCK_N*ii]);
        regB[stage][1] = *reinterpret_cast<float4 *>(&subB[memstage][threadIdx.y * Per_N +4+ BLOCK_N*ii]);

#pragma unroll
        for(int cpi = 0;cpi < Per_M/4;cpi++)
        {
            for(int cpj = 0;cpj < Per_N/4;cpj++)
            {
                c[cpi * 4 * Per_M + cpj * 4] += regA[stage][cpi].x * regB[stage][cpj].x;
                c[cpi * 4 * Per_M + cpj * 4 + 1] += regA[stage][cpi].x * regB[stage][cpj].y;
                c[cpi * 4 * Per_M + cpj * 4 + 2] += regA[stage][cpi].x * regB[stage][cpj].z;
                c[cpi * 4 * Per_M + cpj * 4 + 3] += regA[stage][cpi].x * regB[stage][cpj].w;

                c[(cpi * 4 + 1) * Per_M + cpj * 4] += regA[stage][cpi].y * regB[stage][cpj].x;
                c[(cpi * 4 + 1) * Per_M + cpj * 4 + 1] += regA[stage][cpi].y * regB[stage][cpj].y;
                c[(cpi * 4 + 1) * Per_M + cpj * 4 + 2] += regA[stage][cpi].y * regB[stage][cpj].z;
                c[(cpi * 4 + 1) * Per_M + cpj * 4 + 3] += regA[stage][cpi].y * regB[stage][cpj].w;

                c[(cpi * 4 + 2) * Per_M + cpj * 4] += regA[stage][cpi].z * regB[stage][cpj].x;
                c[(cpi * 4 + 2) * Per_M + cpj * 4 + 1] += regA[stage][cpi].z * regB[stage][cpj].y;
                c[(cpi * 4 + 2) * Per_M + cpj * 4 + 2] += regA[stage][cpi].z * regB[stage][cpj].z;
                c[(cpi * 4 + 2) * Per_M + cpj * 4 + 3] += regA[stage][cpi].z * regB[stage][cpj].w;

                c[(cpi * 4 + 3) * Per_M + cpj * 4] += regA[stage][cpi].w * regB[stage][cpj].x;
                c[(cpi * 4 + 3) * Per_M + cpj * 4 + 1] += regA[stage][cpi].w * regB[stage][cpj].y;
                c[(cpi * 4 + 3) * Per_M + cpj * 4 + 2] += regA[stage][cpi].w * regB[stage][cpj].z;
                c[(cpi * 4 + 3) * Per_M + cpj * 4 + 3] += regA[stage][cpi].w * regB[stage][cpj].w;
            }
        }
        stage ^=1;
    }
    pipeline.consumer_release();

    pipeline.producer_acquire();
#pragma unroll
    for(int i=0;i<Per_M;++i)
    {
        for(int j=0;j<Per_N;j+=4)
        {
            // *reinterpret_cast<float4*>(&baseC[i*N+j]) = *reinterpret_cast<float4*>(&c[i*Per_M+j]);
            cuda::memcpy_async(&c[i*Per_M+j],&baseC[i*N+j],sizeof(float4),pipeline);
        }
    }
    pipeline.producer_commit();
}


void CUDA_Kernel::CUDA_GEMM(float*A,float*B,float*C,int M,int N,int K)
{
    cudaSetDevice(0);
    float*d_a,*d_b,*d_c;
    cudaMallocAsync(&d_a,sizeof(float)*M*K,0);
    cudaMallocAsync(&d_b,sizeof(float)*N*K,0);
    cudaMallocAsync(&d_c,sizeof(float)*N*M,0);

    
    assert(((unsigned long long)d_a) % 128 == 0);
    assert(((unsigned long long)d_b) % 128 == 0);
    assert(((unsigned long long)d_c) % 128 == 0);

    cudaMemcpyAsync(d_a,A,sizeof(float)*M*K,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_b,B,sizeof(float)*N*K,cudaMemcpyHostToDevice,0);

    dim3 block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 grid((M+BLOCK_M-1)/BLOCK_M,(N+BLOCK_N-1)/BLOCK_N);
    GEMM<<<grid,block>>>(d_a,d_b,d_c,M,N,K);
    cudaMemcpyAsync(C,d_c,sizeof(float)*M*N,cudaMemcpyDeviceToHost,0);

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