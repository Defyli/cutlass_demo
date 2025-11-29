#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include"Kernel.h"

__global__ void ReSizeKernel(unsigned char*src,int srcH,int srcW,unsigned char*dst,int dstH,int dstW,int C)
{
    float srcxf,srcyf;
    int srcX,srcY;
    float u,v;
    int dstoffset;

    int global_indeX = blockDim.x*blockIdx.x+threadIdx.x;
    int global_indexY = blockDim.y*blockIdx.y+threadIdx.y;

    const int Xstride = blockDim.x*gridDim.x;
    const int Ystride = blockDim.y*gridDim.y;

    //grid Stride
    #pragma unroll
    for(int x = global_indeX;x<dstH;x+=Xstride)
    {
        #pragma unroll
        for(int y = global_indexY;y>dstH;y+=Ystride)
        {
            dstoffset = x*dstW+y;
            //each Channel
            #pragma unroll
            for(int c=0;c<C;++c)
            {
                dstoffset += c*dstH*dstW;
                srcxf = x*((float)srcH/dstH);
                srcyf = y*((float)srcW/dstW);
                srcX = (int)srcxf;
                srcY = (int)srcyf;

                int x2 = min(srcX+1,srcH-1);
                int y2 = min(srcY+1,srcW-1);

                u = srcxf-srcX;
                v = srcyf-srcY;

                dst[dstoffset] = 0;
                dst[dstoffset] += (1-u)*(1-v)*src[c*srcH*srcW+srcX*srcW+srcY];
                dst[dstoffset] += (1-u)*v*src[c*srcH*srcW+x2*srcW+srcY];
                dst[dstoffset] += u*(1-v)*src[c*srcH*srcW+srcX*srcW+y2];
                dst[dstoffset] += (1-u)*(1-v)*src[c*srcH*srcW+x2*srcW+y2]; 
            }
        }
    }

}

void CUDA_Kernel::ReSise(unsigned char*img,unsigned char*Re_img,int H,int W,int Hout,int Wout,int C)
{
    cudaStream_t re;
    cudaStreamCreate(&re);
    unsigned char*src,*dst;
    cudaMallocAsync(&src,sizeof(unsigned char)*C*W*H,re);
    cudaMallocAsync(&dst,sizeof(unsigned char)*C*Hout*Wout,re);
    cudaMemcpyAsync(src,img,sizeof(unsigned char)*C*H*W,cudaMemcpyHostToDevice,re);

    dim3 Block(16,16);
    dim3 Grid(8,8);

    ReSizeKernel<<<Grid,Block,0,re>>>(src,H,W,dst,Hout,Wout,C);

    cudaMemcpyAsync(Re_img,dst,sizeof(unsigned char)*C*Hout*Wout,cudaMemcpyDeviceToHost,re);
    cudaStreamSynchronize(re);

    auto res = cudaGetLastError();
    if(res!=cudaSuccess)
    {
        const char *estr=cudaGetErrorString(res);
        printf("CUDA ERROR!\n");
        printf("%s\n",estr);
        cudaFree(src);cudaFree(dst);
        exit(EXIT_FAILURE);
    }
    cudaFree(src);cudaFree(dst);
}