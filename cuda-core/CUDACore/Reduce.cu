#include<cuda_runtime.h>
#include<numeric>
#include<algorithm>
#include<iostream>
#include <cooperative_groups.h>
#include "Kernel.h"
namespace cg = cooperative_groups;

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down_sync(0xffffffff,val, offset);
  return val;
}

__device__
int blockReduceSum(float val) {

  __shared__ int shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__global__ void deviceReduceBlockAtomicKernel(float *in, float* out, int N) {
  cg::thread_block block = cg::this_thread_block();
  float sum = int(0);
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; 
      i < N; 
      i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = blockReduceSum(sum);
  if(block.thread_rank()==0)atomicAdd(out, sum);
}


float CUDA_Kernel::ReduceSum(float*A,int N)
{
  cudaSetDevice(0);
  float*d_a;
  float*sum;
  float s = 0;
  cudaMallocAsync(&d_a,sizeof(float)*N,0);
  cudaMallocAsync(&sum,sizeof(float),0);
  cudaMemcpyAsync(d_a,A,sizeof(float)*N,cudaMemcpyHostToDevice);
  cudaMemcpyAsync(sum,&s,sizeof(float),cudaMemcpyHostToDevice);

  cudaError_t e =cudaGetLastError();
  if(e!=cudaSuccess)
  {
      const char *estr=cudaGetErrorString(e);
      printf("CUDA ERROR!\n");
      printf("%s\n",estr);
      exit(EXIT_FAILURE);
  }

  int block = 256;
  int grid = (N+block-1)/block;
  deviceReduceBlockAtomicKernel<<<grid,block>>>(d_a,sum,N);

  cudaMemcpyAsync(&s,sum,sizeof(float),cudaMemcpyDeviceToHost);
  cudaFreeAsync(d_a,0);
  cudaFreeAsync(sum,0);

  cudaDeviceSynchronize();

  e = cudaGetLastError();

  if(e!=cudaSuccess)
  {
      const char *estr=cudaGetErrorString(e);
      printf("CUDA ERROR!\n");
      printf("%s\n",estr);
      exit(EXIT_FAILURE);
  }

  return s;   
}