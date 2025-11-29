#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include<cstdio>
#include<cuda/pipeline>
namespace cg = cooperative_groups;
// Sparse Winograd Knernel Layout 

__device__ __forceinline__ void multiply_G_2x2_3x3(const float in[3],float out[4])
{
    /*
        G = [
                1,    0,   0
                1/2,  1/2, 1/2
                1/2, -1/2, 1/2
                0,    0,   1
            ]
    */

    out[0] = in[0];
    // out[1] = 0.5 * (in[0] + in[1] + in[2]);
    // out[2] = 0.5 * (in[0] - in[1] + in[2]);
    float tmp = in[0] + in[2];
    out[1] = 0.5 * (tmp + in[1]);
    out[2] = 0.5 * (tmp - in[1]);
    out[3] = in[2];
}

__global__ void Transform_Kernel(const float*K,float*U,int Knum)
{
    auto block = cg::this_thread_block();
    const int lane = cg::this_grid().thread_rank();
    const int GridSize = cg::this_grid().num_threads();

    //each thread takes one Kernel
    if(lane<Knum)
    {
        float Gg[4*3];

        for(int i=lane;i<Knum;i+=GridSize)
        {
            //Grid Stride
            float col[3];
            int kernel = i/9;
            //1
            int col_index= 0;
            col[0] = K[kernel*9+col_index];
            col[1] = K[kernel*9+col_index+3];
            col[2] = K[kernel*9+col_index+6];

            float out[4*4];
            multiply_G_2x2_3x3(col,out);
            Gg[0*3+col_index] = out[0];
            Gg[1*3+col_index] = out[1];
            Gg[2*3+col_index] = out[2];
            Gg[3*3+col_index] = out[3];

            //2
            col_index = 1;
            col[0] = K[kernel*9+col_index];
            col[1] = K[kernel*9+col_index+3];
            col[2] = K[kernel*9+col_index+6];
            multiply_G_2x2_3x3(col,out+4);
            Gg[0*3+col_index] = out[4+0];
            Gg[1*3+col_index] = out[4+1];
            Gg[2*3+col_index] = out[4+2];
            Gg[3*3+col_index] = out[4+3];

            //3
            col_index = 2;
            col[0] = K[kernel*9+col_index];
            col[1] = K[kernel*9+col_index+3];
            col[2] = K[kernel*9+col_index+6];
            multiply_G_2x2_3x3(col,out+8);
            Gg[0*3+col_index] = out[8+0];
            Gg[1*3+col_index] = out[8+1];
            Gg[2*3+col_index] = out[8+2];
            Gg[3*3+col_index] = out[8+3];

            block.sync();

            //Gg GT
            //1
            int row_index = 0;
            col[0] = Gg[row_index*3];
            col[1] = Gg[row_index*3+1];
            col[2] = Gg[row_index*3+2];

            multiply_G_2x2_3x3(col,out);

            //async write back
            // cg::memcpy_async(cg::this_thread(),&U[i*4*4],out,sizeof(float)*4);
            *reinterpret_cast<float4*>(&U[i*4*4]) = *reinterpret_cast<float4*>(out);

            //2
            row_index = 1;
            col[0] = Gg[row_index*3];
            col[1] = Gg[row_index*3+1];
            col[2] = Gg[row_index*3+2];

            multiply_G_2x2_3x3(col,out+4);

            //async write back
            // cg::memcpy_async(cg::this_thread(),&U[i*4*4],out+4,sizeof(float)*4);
            *reinterpret_cast<float4*>(&U[i*4*4+4]) = *reinterpret_cast<float4*>(out+4);

            //3
            row_index = 2;
            col[0] = Gg[row_index*3];
            col[1] = Gg[row_index*3+1];
            col[2] = Gg[row_index*3+2];

            multiply_G_2x2_3x3(col,out+8);

            //async write back
            // cg::memcpy_async(cg::this_thread(),&U[i*4*4],out+8,sizeof(float)*4);
            *reinterpret_cast<float4*>(&U[i*4*4+8]) = *reinterpret_cast<float4*>(out+8);

            //4
            row_index = 3;
            col[0] = Gg[row_index*3];
            col[1] = Gg[row_index*3+1];
            col[2] = Gg[row_index*3+2];

            multiply_G_2x2_3x3(col,out+12);

            //async write back
            // cg::memcpy_async(cg::this_thread(),&U[i*4*4],out+12,sizeof(float)*4);
            *reinterpret_cast<float4*>(&U[i*4*4+12]) = *reinterpret_cast<float4*>(out+12);

            
            // cg::wait(cg::this_thread());

        }
    }
}

__device__ __forceinline__ void multiply_BT_2x2_3x3(const float in[4], float out[4])
{
    /*
        B = [
            1, 0, -1, 0
            0, 1, 1, 0
            0, -1, 1, 0
            0, 1, 0, -1
        ]
    */

    out[0] = in[0] - in[2];
    out[1] = in[1] + in[2];
    out[2] = in[2] - in[1];
    out[3] = in[1] - in[3];
}


// Thread Tile (16,16)*(x,y)
// H,W is the height after padding
// Note that beside padding, we also need Row-padding and column-padding to ensure that the feature can be sliced into 4*4 pieces
// specificly we need to add one more column in the right-end of the feature if W%2==1 and add one more row if H%2==1
__global__ void winograd_2x2_3x3_BTdB(const float*f,const int C,const int H,const int W,float*d_img)
{
    //each thread transforms one input winow
    auto global_index = cg::this_grid().thread_rank();
    const int Row_idx = blockIdx.y*blockDim.y+threadIdx.y;
    const int Col_idx = blockIdx.x*blockDim.x+threadIdx.x;

    #pragma unroll
    for(int i=Row_idx;i*4+3<C*H;i+=gridDim.y*blockDim.y)
    {
        #pragma unroll
        for(int j=Col_idx;j*4+3<W;j+=gridDim.x*blockDim.x)
        {

            float data[4*4];

            // fetch from global mem
            *reinterpret_cast<float4*>(data) = *reinterpret_cast<const float4*>(f+i*4*W+j*4);
            *reinterpret_cast<float4*>(data+4) = *reinterpret_cast<const float4*>(f+(i*4+1)*W+j*4);
            *reinterpret_cast<float4*>(data+8) = *reinterpret_cast<const float4*>(f+(i*4+2)*W+j*4);
            *reinterpret_cast<float4*>(data+12) = *reinterpret_cast<const float4*>(f+(i*4+3)*W+j*4);
            
            //Bd
            float Bd[4*4];
            float col[4],out[4*4];
            col[0] = data[0+0];
            col[1] = data[4+0];
            col[2] = data[8+0];
            col[3] = data[12+0];
            multiply_BT_2x2_3x3(col,out);
            Bd[0+0] = out[0];
            Bd[4+0] = out[1];
            Bd[8+0] = out[2];
            Bd[12+0] = out[3];

            col[0] = data[0+1];
            col[1] = data[4+1];
            col[2] = data[8+1];
            col[3] = data[12+1];
            multiply_BT_2x2_3x3(col,out);
            Bd[0+1] = out[0];
            Bd[4+1] = out[1];
            Bd[8+1] = out[2];
            Bd[12+1] = out[3];


            col[0] = data[0+2];
            col[1] = data[4+2];
            col[2] = data[8+2];
            col[3] = data[12+2];
            multiply_BT_2x2_3x3(col,out);
            Bd[0+2] = out[0];
            Bd[4+2] = out[1];
            Bd[8+2] = out[2];
            Bd[12+2] = out[3];


            col[0] = data[0+3];
            col[1] = data[4+3];
            col[2] = data[8+3];
            col[3] = data[12+3];
            multiply_BT_2x2_3x3(col,out);
            Bd[0+3] = out[0];
            Bd[4+3] = out[1];
            Bd[8+3] = out[2];
            Bd[12+3] = out[3];

            //BdBt
            col[0] = Bd[0+1];
            col[1] = Bd[0+1];
            col[2] = Bd[0+2];
            col[3] = Bd[0+3];
            multiply_BT_2x2_3x3(col,out);
            // cg::memcpy_async(cg::this_thread(),d_img+global_index*16,out,sizeof(float)*4); // To accelerate memory accessing the output stores like [4*4,4*4,...]
            *reinterpret_cast<float4*>(d_img+global_index*16) = *reinterpret_cast<float4*>(out);
        

            col[0] = Bd[4+1];
            col[1] = Bd[4+1];
            col[2] = Bd[4+2];
            col[3] = Bd[4+3];
            multiply_BT_2x2_3x3(col,out+4);
            // cg::memcpy_async(cg::this_thread(),d_img+global_index*16+4,out+4,sizeof(float)*4);
            *reinterpret_cast<float4*>(d_img+global_index*16+4) = *reinterpret_cast<float4*>(out+4);

            
            col[0] = Bd[8+1];
            col[1] = Bd[8+1];
            col[2] = Bd[8+2];
            col[3] = Bd[8+3];
            multiply_BT_2x2_3x3(col,out+8);
            // cg::memcpy_async(cg::this_thread(),d_img+global_index*16+8,out+8,sizeof(float)*4);
            *reinterpret_cast<float4*>(d_img+global_index*16+8) = *reinterpret_cast<float4*>(out+8);



            col[0] = Bd[12+1];
            col[1] = Bd[12+1];
            col[2] = Bd[12+2];
            col[3] = Bd[12+3];
            multiply_BT_2x2_3x3(col,out+12);
            // cg::memcpy_async(cg::this_thread(),d_img+global_index*16+12,out+12,sizeof(float)*4);
            *reinterpret_cast<float4*>(d_img+global_index*16+12) = *reinterpret_cast<float4*>(out+12);

            // cg::wait(cg::this_thread());

        }
        
    }
}


__device__ __forceinline__ void multiply_AT_2x2_3x3(const float in[4], float out[2])
{
    out[0] = in[0] + in[1] + in[2];
    out[1] = in[1] - (in[2] + in[3]);
}

// In MM we attach each Output Channel to one thread block
// And we fuse the MM with the AMAt
// You should ensure the thread block can not be overcame Sharad memory size
// 512*2*16 = 16K 16K*4B = 64KB
// To ensure the corectness when the tile num is more than 512, we use gride method, which should be arranged well by setting the dim(x,y)
template<int Threads>
__global__ void __launch_bounds__(512,1) ElemetWise_MM_FusedAta(const float*d_k,const float*d_img,const int C,const int N,const int H,const int W,const int*Kernel_idx,float*M,const int Knum)
{
    const int OutC = blockIdx.x;
    const int lane = cg::this_thread_block().thread_rank();
    const int TileX = (W+3)/4;
    const int TileY = (H+3)/4;
    const int lanex = threadIdx.x;
    const int laney = threadIdx.y;
    const int OutRow = TileY*2;
    const int OutCol = TileX*2;
    int Kindx = 0;
    for(;Kindx<Knum;++Kindx)
        if(Kernel_idx[Kindx]>=OutC*C)
            break;

    __shared__ float SubImg[2][Threads*16]; //double buffer
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block,2>share_state;
    float SubOut[4*4];
    for(int i=0;i<16;++i)SubOut[i]=0;
    int stage = 0;
    auto block = cg::this_thread_block();
    auto pipeline = cuda::make_pipeline(block,&share_state);
    
    //to ensure meemory access efficiency we have to store 16 elements in one bank
    pipeline.producer_acquire();
    #pragma unroll
    for(int k=0;k<Threads*16;k+=Threads)
    {
        const int tile = k/16;
        const int tile_index = k%16;
        const int memrow = tile/blockDim.x;
        const int memcol = tile%blockDim.x;
        // SubImg[stage][tile+tile_index*32] = d_img[+memrow*TileX*16+memcol+TileX*TileY*16];
        cuda::memcpy_async(block,&SubImg[stage][tile+tile_index*32],&d_img[memrow*TileX*16+memcol*16+tile_index],sizeof(float),pipeline);
    }
    // __syncthreads();
    pipeline.producer_commit();

    float K[4*4];
    //gride Stride
    #pragma unroll
    for(int i=0;i<TileY;i+=blockDim.y)
    {
        #pragma unroll
        for(int j=0;j<TileX;j+=blockDim.x)
        {
            for(int InputC = Kernel_idx[Kindx]%C;InputC<(OutC+1)*C&&Kindx<Knum;++Kindx)
            {
                //Prefetch
                pipeline.producer_acquire();
                #pragma unroll
                for(int k=0;k<Threads*16;k+=Threads)
                {
                    const int tile = k/16;
                    const int tile_index = k%16;
                    const int memrow = tile/blockDim.x;
                    const int memcol = tile%blockDim.x;
                    // SubImg[stage^1][tile+tile_index*32] = d_img[(i*4+memrow*TileX)*16+memcol+j*4+InputC*TileX*TileY*16];
                    cuda::memcpy_async(block,&SubImg[stage^1][tile+tile_index*32],&d_img[((i*4+memrow)*TileX)*16+(memcol+j*4)*16+InputC*TileX*TileY*16+tile_index],sizeof(float),pipeline);
                }
                // // __syncthreads();
                pipeline.producer_commit();

                *reinterpret_cast<float4*>(K) = *reinterpret_cast<const float4*>(d_k+Kindx*16);
                *reinterpret_cast<float4*>(K+4) = *reinterpret_cast<const float4*>(d_k+Kindx*16+4);
                *reinterpret_cast<float4*>(K+8) = *reinterpret_cast<const float4*>(d_k+Kindx*16+8);
                *reinterpret_cast<float4*>(K+12) = *reinterpret_cast<const float4*>(d_k+Kindx*16+12);

                pipeline.consumer_wait();
                // elementwise compute
                #pragma unroll
                for(int z=0;z<16;++z)
                    SubOut[z] += SubImg[stage][lane+i*32]*K[z];
                stage ^= 1;
                pipeline.consumer_release();
            }

            pipeline.consumer_wait();
            #pragma unroll
            for(int z=0;z<16;++z)
                SubOut[z] += SubImg[stage][lane+z*32]*K[z];
            pipeline.consumer_release();

            //Do AtA
            float AtO[2*4];
            float SubO[2*2];
            AtO[0] = SubOut[0]+SubOut[0+4]+SubOut[0+8];
            AtO[0+4] = SubOut[0+4]-(SubOut[0+8]+SubOut[0+12]);

            AtO[1] = SubOut[1]+SubOut[1+4]+SubOut[1+8];
            AtO[1+4] = SubOut[1+4]-(SubOut[1+8]+SubOut[1+12]);

            AtO[2] = SubOut[2]+SubOut[2+4]+SubOut[2+8];
            AtO[2+4] = SubOut[2+4]-(SubOut[2+8]+SubOut[2+12]);

            AtO[3] = SubOut[3]+SubOut[3+4]+SubOut[3+8];
            AtO[3+4] = SubOut[2+4]-(SubOut[3+8]+SubOut[3+12]);

            SubO[0] = AtO[0]+AtO[1]+AtO[2];
            SubO[1] = AtO[1]-(AtO[2]+AtO[3]);
            SubO[2] = AtO[0+4]+AtO[1+4]+AtO[2+4];
            SubO[3] = AtO[1+4]-(AtO[2+4]+AtO[3+4]);

            // we have to keep ensure that the write is row-major
            const int WriteBase = OutC*OutRow*OutCol+i*2+j*2;
            M[WriteBase] = SubO[0];
            M[WriteBase+1] = SubO[1];
            M[WriteBase+OutCol] = SubO[2];
            M[WriteBase+OutCol+1] = SubO[3];
        }
    }
}

void WinogradSparseF23(float*K,float*Img,int N,int C,int H,int W,float*O,int*Kindex,int Knum)
{
    cudaStream_t imgTrans,kTrans,MM;
    cudaStreamCreate(&imgTrans);
    cudaStreamCreate(&kTrans);
    cudaStreamCreate(&MM);
    float*d_k,*d_f,*d_U,*d_img,*d_o;
    int*d_index;
    cudaMallocAsync(&d_k,sizeof(float)*Knum*9,kTrans);
    cudaMallocAsync(&d_U,sizeof(float)*16*Knum,kTrans);
    cudaMemcpyAsync(d_k,K,sizeof(float)*Knum*9,cudaMemcpyHostToDevice,kTrans);
    dim3 blockTransK(256);
    dim3 gridTransK((Knum+255)/256);
    Transform_Kernel<<<gridTransK,blockTransK,0,kTrans>>>(d_k,d_U,Knum);

    cudaMallocAsync(&d_f,sizeof(float)*C*H*W,imgTrans);
    cudaMallocAsync(&d_img,sizeof(float)*C*((H+3)/4*(W+3)/4),imgTrans);
    cudaMemcpyAsync(d_f,Img,sizeof(float)*C*H*W,cudaMemcpyHostToDevice,imgTrans);
    dim3 blcokBtB(16,16);
    dim3 gridBtb(2,2);
    winograd_2x2_3x3_BTdB<<<gridBtb,blcokBtB,0,imgTrans>>>(d_f,C,H,W,d_img);


    cudaMallocAsync(&d_index,sizeof(int)*Knum,MM);
    cudaMallocAsync(&d_o,sizeof(float)*N*4*((H+3)/4*(W+3)/4),MM);

    cudaStreamSynchronize(imgTrans);
    cudaStreamSynchronize(kTrans);

    ElemetWise_MM_FusedAta<256><<<N,blcokBtB,0,MM>>>(d_k,d_img,C,N,H,W,d_index,d_o,Knum);

    cudaStreamSynchronize(MM);

    cudaError_t e = cudaGetLastError();
    if(e!=cudaSuccess)
    {
        const char *estr=cudaGetErrorString(e);
        cudaFree(d_k);cudaFree(d_f);cudaFree(d_U);cudaFree(d_img);cudaFree(d_o);cudaFree(d_index);
        printf("CUDA ERROR!\n");
        printf("%s\n",estr);
        exit(EXIT_FAILURE); 
    }
    cudaMemcpy(O,d_o,sizeof(float)*N*4*((H+1)/2*(W+1)/2),cudaMemcpyDeviceToHost);
    cudaFree(d_k);cudaFree(d_f);cudaFree(d_U);cudaFree(d_img);cudaFree(d_o);cudaFree(d_index);
}