#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include "Kernel.h"
namespace cg = cooperative_groups;
__device__ __forceinline__ float SUM_FOUR(float a,float b,float c,float d)
{
    
    // asm("vadd4.f32.f32.f32.f32" "%0,%1,%2,%3;":"+f"(a):"f"(b),"f"(c),"f"(d));
    // return a;
    return a+b+c+d;
}


__global__ void SUM_VEC(float*v,float*sum,size_t n)
{
    const unsigned int threadid = cg::this_thread_block().thread_rank();
    const unsigned int warpid  = threadid/warpSize;
    const int warpsize = cg::this_thread_block().size()/warpSize;
    __shared__ float temp[32];
    if(threadid*4<n)
    {
        float val = 0;
        for(int offset=threadid*4;offset<n;offset+=cg::this_thread_block().size()*4)
        {
            val += SUM_FOUR(v[threadid*4],v[threadid*4+1],v[threadid*4+2],v[threadid*4+3]);
        }

        //reduce
        for(int offset = warpSize/2;offset>0;offset/=2)
            val += __shfl_down_sync(0xffffffff,val,offset);

        if(threadid==0)temp[warpid] = val;

        __syncthreads();
        
        if(threadid<warpSize)val = temp[threadid];
        else val = 0;

        __syncthreads();

        for(int offset = warpSize/2;offset>0;offset/=2)
            val += __shfl_down_sync(0xffffffff,val,offset);
        
        if(threadid==0)*sum = val;
        
    }
}

float CUDA_Kernel::SIMD_Sum(float*v,size_t n)
{
    float*d_v,*sum;
    float res;
    cudaMallocAsync(&d_v,sizeof(float)*n,0);
    cudaMallocAsync(&sum,sizeof(float),0);
    cudaMemcpyAsync(d_v,v,sizeof(float)*n,cudaMemcpyHostToDevice,0);
    dim3 Block(1024);

    SUM_VEC<<<1,Block>>>(v,sum,n);

    cudaMemcpyAsync(&res,sum,sizeof(float),cudaMemcpyDeviceToDevice,0);

    cudaFreeAsync(d_v,0);
    cudaFreeAsync(sum,0);

    cudaStreamSynchronize(0);

    return res;
}