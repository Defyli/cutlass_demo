#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include "Kernel.h"
namespace cg = cooperative_groups;
__device__ __forceinline__ void MereSort(float*data,int n,float*temp)
{
    int p1 = 0;
    int p2 = p1+n/2;
    int p = 0;
    while(p1<n/2&&p2<n)
    {
        if(data[p1]>data[p2])temp[p++] = data[p2++];
        else temp[p++] = data[p1++]; 
    }
    while(p1<n/2)temp[p++] = data[p1++];
    while(p2<n)temp[p++] = data[p2++];

    //copy back
    for(int i=0;i<n;++i)
        data[i] = temp[i];
}

__global__ void MeregeSort(float*A,int n,float*temp)
{
    auto lane = cg::this_grid().thread_rank();
    const int total = cg::this_grid().num_threads();

    int stride;
    for(stride = 2;stride<=n;stride*=2)
    {
        for(int i = lane*stride;i<n;i+=total*stride)
        {
            MereSort(A+i,min(stride,n-i+1),temp+i);
        }
    }

    if(n%2!=0)
    {
        if(lane==0)
        {
            //last merege
            stride /=2;
            int p1=0;
            int p2=stride;
            int p=0;
            while(p1<stride&&p2<n)
            {
                if(A[p1]>A[p2])temp[p++] = A[p2++];
                else temp[p++] = A[p1++]; 
            }
            while(p1<stride)temp[p++] = A[p1++];
            while(p2<n)temp[p++] = A[p2++];
        }
        //copy back
        for(int i=lane;i<n;i+=total)
            A[i] = temp[i];
    }
}


void CUDA_Kernel::MeregeSortGPU(float*data,int n)
{
    cudaStream_t sort;
    cudaStreamCreate(&sort);
    float*d,*d_temp;
    cudaMallocAsync(&d,sizeof(float)*n,sort);
    cudaMallocAsync(&d_temp,sizeof(float)*n,sort);
    cudaMemcpyAsync(d,data,sizeof(float)*n,cudaMemcpyHostToDevice,sort);
    
    dim3 Block(256);
    dim3 Grid(1);
    MeregeSort<<<Grid,Block,0,sort>>>(d,n,d_temp);

    cudaStreamSynchronize(sort);

    cudaError_t e = cudaGetLastError();
    if(e!=cudaSuccess)
    {
        const char *estr=cudaGetErrorString(e);
        cudaFree(d);cudaFree(d_temp);
        printf("CUDA ERROR!\n");
        printf("%s\n",estr);
        exit(EXIT_FAILURE); 
    }

    cudaMemcpy(data,d,sizeof(float)*n,cudaMemcpyDeviceToHost);
    cudaFree(d);cudaFree(d_temp);
}