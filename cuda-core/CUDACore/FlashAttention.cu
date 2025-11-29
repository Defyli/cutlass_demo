#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include <cuda/pipeline>
#include<cmath>
#include<algorithm>
#include<float.h>
#include"Kernel.h"
#define warpSize 32
namespace cg = cooperative_groups;
// constexpr int d = 1024;  //1024 encode length: more than 16 //changeable

constexpr int Bk = 32; // to avoid bank conficit more than 32
// constexpr int Kd = (d+Bk-1)/d;
constexpr int Bc = 32; 
constexpr int Br = 32;

constexpr int ThreadPerBlock = 256;
constexpr int WarpBlock = 8;
constexpr int MAX_REGIER_NUMS = 96;


constexpr int SubOr = Br/WarpBlock;
constexpr int SubOc = Bk/warpSize;
// constexpr int EachThreadHoldO = Kd*SubOc*SubOr;  // 由于(Bc,d)的输出O维度可能比较大，计算时采用d维度切分（非并行），因而存储采用block-stride多存储

__device__ __forceinline__ float warpReduceSum(float val) {
  #pragma unroll
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down_sync(0xffffffff,val, offset);
  return val;
}

__device__ __forceinline__ float RowMax(float*data,int n)
{
    float val = FLT_MIN;
    int n_temp = n+(n-n%warpSize);
    for(int i=0;i<n_temp;i+=warpSize)
    {
        if(i<n)val = fmaxf(val,data[i]);
        #pragma unroll 
        for (int mask = warpSize >> 1; mask >= 1; mask >>= 1) {
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
        }
    }
    return val;
}

__device__ __forceinline__ float RowSum(float*data,int n)
{
    float val = 0;
    for(int i=0;i<n;i+=warpSize)
    {
        val += data[i];
    }
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xffffffff,val, offset);
    return val;
}
__global__ void FlashAttention_16_FullSeq(float*Q,float*K,float*V,float*O,size_t N,size_t d)
{
    auto block = cg::this_thread_block();
    const int threadid = block.thread_rank();
    const int warpid = threadid/warpSize;
    const int lane = threadid%warpSize;
    const int Kd = (d+Bk-1)/Bk;
    // const int EachThreadHoldO = Kd*SubOc*SubOr; 

    constexpr auto scope = cuda::thread_scope_block;
    constexpr auto stages_count = 2;
    __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
    auto pipeline = cuda::make_pipeline(block,&shared_state);

    __shared__ float P[Br*Bc];
    __shared__ float SubQ[2][Br*Bk];
    __shared__ float SubKV[2][Bk*Bc];  //K V shared
    // float *SubO = new float[EachThreadHoldO];
    float SubO[MAX_REGIER_NUMS]={0};

    int Row = blockIdx.x*Br;// each block compute one
    int stage = 0;

    __shared__ float rowmax[Br];
    if(threadid<Br) rowmax[threadid] = FLT_MIN;
    __shared__ float L[Br];
    if(threadid<Br) L[threadid] = 0;

    block.sync();


    for(int j=0;j<N;j+=Bc)
    {
        // load and compute
        pipeline.producer_acquire();
        #pragma unroll
        for(int LoadQr = warpid; LoadQr<Br;LoadQr+=WarpBlock)
            for(int LoadQc = lane;LoadQc<Bk;LoadQc+=warpSize)
                cuda::memcpy_async(block,&SubQ[stage][LoadQr*Bk+LoadQc],&Q[(Row+LoadQr)*d+LoadQc],sizeof(float),pipeline);
        
        // Transpose B
        #pragma unroll 
        for(int LoadKc = warpid;LoadKc<Bc;LoadKc+=WarpBlock)
            for(int LoadKr = lane;LoadKc<Bk;LoadKc+=warpSize)
                cuda::memcpy_async(block,&SubKV[stage][LoadKr+LoadKc*Bk],&K[LoadKc+j+LoadKr*N],sizeof(float),pipeline);
        
        pipeline.producer_commit();
        // Compute Pij
        for(int k = Bk;k<d;k+=Bk)
        {           
 
            pipeline.producer_acquire();
            #pragma unroll
            for(int LoadQr = warpid; LoadQr<Br;LoadQr+=WarpBlock)
                for(int LoadQc = lane;LoadQc<Bk;LoadQc+=warpSize)
                    cuda::memcpy_async(block,&SubQ[stage^1][LoadQr*Bk+LoadQc],&Q[(Row+LoadQr)*d+LoadQc+k],sizeof(float),pipeline);
            
            // Transpose B
            #pragma unroll 
            for(int LoadKc = warpid;LoadKc<Bc;LoadKc+=WarpBlock)
                for(int LoadKr = lane;LoadKc<Bk;LoadKc+=warpSize)
                    cuda::memcpy_async(block,&SubKV[stage^1][LoadKr+LoadKc*Bk],&K[LoadKc+j+(LoadKr+k)*N],sizeof(float),pipeline);

            pipeline.producer_commit();

            pipeline.consumer_wait();

            #pragma unroll
            for(int Psubi = warpid;Psubi<Br;Psubi+=WarpBlock)
            {
                #pragma unroll
                for(int Psubj=0;Psubj<Bc;++Psubj)
                {
                    float val = SubQ[stage][Psubi*Bk+lane]*SubKV[stage][Psubj*Bk+lane];
                    __syncwarp();
                    val = warpReduceSum(val);
                    if(lane==0)P[Psubi*Bc+Psubj] = val;
                }
            }
            pipeline.consumer_release();
            stage ^=1;

        }
        pipeline.consumer_wait();

        #pragma unroll
        for(int Psubi = warpid;Psubi<Br;Psubi+=WarpBlock)
        {
            #pragma unroll
            for(int Psubj=0;Psubj<Bc;++Psubj)
            {
                float val = SubQ[stage][Psubi*Bk+lane]*SubKV[stage][Psubj*Bk+lane];
                __syncwarp();
                val = warpReduceSum(val);
                if(lane==0)P[Psubi*Bc+Psubj] = val;
            }
        }
        pipeline.consumer_release();
        

        //Update M and L
        float temp;
        #pragma unroll
        for(int mi = warpid;mi<Br;mi+=WarpBlock)
        {
            float max = RowMax(&P[mi*Bc],Bc);
            temp = rowmax[mi];
            if(lane==0)rowmax[mi] = fmax(max,P[mi]);
        }
        __syncwarp();


        #pragma unroll
        for(int li = warpid;li<Br;li+=WarpBlock)
        {
            #pragma unroll
            for(int lj = lane;lj<Bc;lj+=warpSize)
                P[li*Bc+lj] = expf(P[li*Bc+lj]-rowmax[li]);
        }

        #pragma unroll
        for(int li = warpid;li<Br;li+=WarpBlock)
        {
            float sum = RowSum(&P[li*Bc],Bc);
            temp = expf(temp-rowmax[li]);
            L[li] = temp*L[li]+sum;
        }

        //update O
        stage = 0;
        pipeline.producer_acquire();
        #pragma unroll
        for(int LoadVc=warpid;LoadVc<Bk;LoadVc+=WarpBlock)
            for(int LoadVr=lane;LoadVr<Br;LoadVr+=warpSize)
                cuda::memcpy_async(block,&SubKV[stage][LoadVc*Br+LoadVr],&V[(Row+LoadVr)*d+LoadVc],sizeof(float),pipeline);
        pipeline.producer_commit();


        for(int k=Bk;k<d;k+=Bk)
        {
            pipeline.producer_acquire();
            #pragma unroll
            for(int LoadVc=warpid;LoadVc<Bk;LoadVc+=WarpBlock)
                for(int LoadVr=lane;LoadVr<Br;LoadVr+=warpSize)
                    cuda::memcpy_async(block,&SubKV[stage^1][LoadVc*Br+LoadVr],&V[(Row+LoadVr)*d+LoadVc+k],sizeof(float),pipeline);
            pipeline.producer_commit();


            pipeline.consumer_wait();
            int suboffset = (k/Bk-1)*SubOc*SubOr;
            #pragma unroll
            for(int Oi = warpid;Oi<Br;Oi+=WarpBlock)
            {
                for(int Oj = 0;Oj<Bk;++Oj)
                {
                    int SubOi = Oi/WarpBlock;
                    int SubOj = Oj/warpSize;
                    __syncwarp();
                    float val = warpReduceSum(P[Oi*Bc+lane]*SubKV[stage][Oj*Bc+lane]);
                    __shfl_up_sync(0xffffffff,val,Oj%warpSize); //fetch data from tid 0;
                    if(lane==Oj)SubO[suboffset+SubOi*SubOc+SubOj] = SubO[suboffset+SubOi*SubOc+SubOj]*temp + val;
                }
            }
            pipeline.consumer_release();

            stage ^=1;
        }

        pipeline.consumer_wait();
        int suboffset = (Kd-1)*SubOc*SubOr;
        #pragma unroll
        for(int Oi = warpid;Oi<Br;Oi+=WarpBlock)
        {
            for(int Oj = 0;Oj<Bk;++Oj)
            {
                int SubOi = Oi/WarpBlock;
                int SubOj = Oj/warpSize;
                __syncwarp();
                float val = warpReduceSum(P[Oi*Bc+lane]*SubKV[stage][Oj*Bc+lane]);
                val = __shfl_up_sync(0xffffffff,val,Oj%warpSize); //fetch data from tid 0;
                if(lane==Oj)SubO[suboffset+SubOi*SubOc+SubOj] = SubO[suboffset+SubOi*SubOc+SubOj]*temp + val;
            }
        }
        pipeline.consumer_release();
    }



    // write back
    #pragma unroll
    for(int OK=0;OK<Kd;++OK)
    {
        #pragma unroll
        for(int Oi=0;Oi<SubOr;++Oi)
        {
            #pragma unroll
            for(int Oj=0;Oj<SubOc;++Oj)
                O[(Row+Oi*WarpBlock)*d+OK*Bk+lane+Oj/warpSize] += SubO[Oi*SubOc+Oj+OK*SubOr*SubOc];
        }
    }
}

// Sparse Requires the Br==Bc to keep window sparse format
__global__ void SparseWindowAttention(float*Q,float*K,float*V,float*O,size_t N,size_t d,size_t window)
{
    auto block = cg::this_thread_block();
    const int threadid = block.thread_rank();
    const int warpid = threadid/warpSize;
    const int lane = threadid%warpSize;
    const int Kd = (d+Bk-1)/Bk;
    // const int EachThreadHoldO = Kd*SubOc*SubOr; 

    constexpr auto scope = cuda::thread_scope_block;
    constexpr auto stages_count = 2;
    __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
    auto pipeline = cuda::make_pipeline(block,&shared_state);

    __shared__ float P[Br*Bc];
    __shared__ float SubQ[2][Br*Bk];
    __shared__ float SubKV[2][Bk*Bc];  //K V shared
    // float *SubO = new float[EachThreadHoldO];
    float SubO[MAX_REGIER_NUMS]={0};

    int Row = blockIdx.x*Br;// each block compute one
    int stage = 0;

    __shared__ float rowmax[Br];
    if(threadid<Br) rowmax[threadid] = FLT_MIN;
    __shared__ float L[Br];
    if(threadid<Br) L[threadid] = 0;

    block.sync();


    for(int j=Row;j<min(N,Row+Bc*window);j+=Bc)
    {
        // load and compute
        pipeline.producer_acquire();
        #pragma unroll
        for(int LoadQr = warpid; LoadQr<Br;LoadQr+=WarpBlock)
            for(int LoadQc = lane;LoadQc<Bk;LoadQc+=warpSize)
                cuda::memcpy_async(block,&SubQ[stage][LoadQr*Bk+LoadQc],&Q[(Row+LoadQr)*d+LoadQc],sizeof(float),pipeline);
        
        // Transpose B
        #pragma unroll 
        for(int LoadKc = warpid;LoadKc<Bc;LoadKc+=WarpBlock)
            for(int LoadKr = lane;LoadKc<Bk;LoadKc+=warpSize)
                cuda::memcpy_async(block,&SubKV[stage][LoadKr+LoadKc*Bk],&K[LoadKc+j+LoadKr*N],sizeof(float),pipeline);
        
        pipeline.producer_commit();
        // Compute Pij
        for(int k = Bk;k<d;k+=Bk)
        {           
 
            pipeline.producer_acquire();
            #pragma unroll
            for(int LoadQr = warpid; LoadQr<Br;LoadQr+=WarpBlock)
                for(int LoadQc = lane;LoadQc<Bk;LoadQc+=warpSize)
                    cuda::memcpy_async(block,&SubQ[stage^1][LoadQr*Bk+LoadQc],&Q[(Row+LoadQr)*d+LoadQc+k],sizeof(float),pipeline);
            
            // Transpose B
            #pragma unroll 
            for(int LoadKc = warpid;LoadKc<Bc;LoadKc+=WarpBlock)
                for(int LoadKr = lane;LoadKc<Bk;LoadKc+=warpSize)
                    cuda::memcpy_async(block,&SubKV[stage^1][LoadKr+LoadKc*Bk],&K[LoadKc+j+(LoadKr+k)*N],sizeof(float),pipeline);

            pipeline.producer_commit();

            pipeline.consumer_wait();

            #pragma unroll
            for(int Psubi = warpid;Psubi<Br;Psubi+=WarpBlock)
            {
                #pragma unroll
                for(int Psubj=0;Psubj<Bc;++Psubj)
                {
                    float val = SubQ[stage][Psubi*Bk+lane]*SubKV[stage][Psubj*Bk+lane];
                    __syncwarp();
                    val = warpReduceSum(val);
                    if(lane==0)P[Psubi*Bc+Psubj] = val;
                }
            }
            pipeline.consumer_release();
            stage ^=1;

        }
        pipeline.consumer_wait();

        #pragma unroll
        for(int Psubi = warpid;Psubi<Br;Psubi+=WarpBlock)
        {
            #pragma unroll
            for(int Psubj=0;Psubj<Bc;++Psubj)
            {
                float val = SubQ[stage][Psubi*Bk+lane]*SubKV[stage][Psubj*Bk+lane];
                __syncwarp();
                val = warpReduceSum(val);
                if(lane==0)P[Psubi*Bc+Psubj] = val;
            }
        }
        pipeline.consumer_release();
        

        //Update M and L
        float temp;
        #pragma unroll
        for(int mi = warpid;mi<Br;mi+=WarpBlock)
        {
            float max = RowMax(&P[mi*Bc],Bc);
            temp = rowmax[mi];
            if(lane==0)rowmax[mi] = fmax(max,P[mi]);
        }
        __syncwarp();


        #pragma unroll
        for(int li = warpid;li<Br;li+=WarpBlock)
        {
            #pragma unroll
            for(int lj = lane;lj<Bc;lj+=warpSize)
                P[li*Bc+lj] = expf(P[li*Bc+lj]-rowmax[li]);
        }

        #pragma unroll
        for(int li = warpid;li<Br;li+=WarpBlock)
        {
            float sum = RowSum(&P[li*Bc],Bc);
            temp = expf(temp-rowmax[li]);
            L[li] = temp*L[li]+sum;
        }

        //update O
        stage = 0;
        pipeline.producer_acquire();
        #pragma unroll
        for(int LoadVc=warpid;LoadVc<Bk;LoadVc+=WarpBlock)
            for(int LoadVr=lane;LoadVr<Br;LoadVr+=warpSize)
                cuda::memcpy_async(block,&SubKV[stage][LoadVc*Br+LoadVr],&V[(Row+LoadVr)*d+LoadVc],sizeof(float),pipeline);
        pipeline.producer_commit();


        for(int k=Bk;k<d;k+=Bk)
        {
            pipeline.producer_acquire();
            #pragma unroll
            for(int LoadVc=warpid;LoadVc<Bk;LoadVc+=WarpBlock)
                for(int LoadVr=lane;LoadVr<Br;LoadVr+=warpSize)
                    cuda::memcpy_async(block,&SubKV[stage^1][LoadVc*Br+LoadVr],&V[(Row+LoadVr)*d+LoadVc+k],sizeof(float),pipeline);
            pipeline.producer_commit();


            pipeline.consumer_wait();
            int suboffset = (k/Bk-1)*SubOc*SubOr;
            #pragma unroll
            for(int Oi = warpid;Oi<Br;Oi+=WarpBlock)
            {
                for(int Oj = 0;Oj<Bk;++Oj)
                {
                    int SubOi = Oi/WarpBlock;
                    int SubOj = Oj/warpSize;
                    __syncwarp();
                    float val = warpReduceSum(P[Oi*Bc+lane]*SubKV[stage][Oj*Bc+lane]);
                    __shfl_up_sync(0xffffffff,val,Oj%warpSize); //fetch data from tid 0;
                    if(lane==Oj)SubO[suboffset+SubOi*SubOc+SubOj] = SubO[suboffset+SubOi*SubOc+SubOj]*temp + val;
                }
            }
            pipeline.consumer_release();

            stage ^=1;
        }

        pipeline.consumer_wait();
        int suboffset = (Kd-1)*SubOc*SubOr;
        #pragma unroll
        for(int Oi = warpid;Oi<Br;Oi+=WarpBlock)
        {
            for(int Oj = 0;Oj<Bk;++Oj)
            {
                int SubOi = Oi/WarpBlock;
                int SubOj = Oj/warpSize;
                __syncwarp();
                float val = warpReduceSum(P[Oi*Bc+lane]*SubKV[stage][Oj*Bc+lane]);
                val = __shfl_up_sync(0xffffffff,val,Oj%warpSize); //fetch data from tid 0;
                if(lane==Oj)
                {
                    SubO[suboffset+SubOi*SubOc+SubOj] = SubO[suboffset+SubOi*SubOc+SubOj]*temp + val;
                    // window resdiual
                    if(j+Bc>min(N,Bc*window))SubO[suboffset+SubOi*SubOc+SubOj] = SubO[suboffset+SubOi*SubOc+SubOj]*rowmax[Oi]+ (N-j)/Bc;
                }
            }
        }
        pipeline.consumer_release();
    }

    // write back
    #pragma unroll
    for(int OK=0;OK<Kd;++OK)
    {
        #pragma unroll
        for(int Oi=0;Oi<SubOr;++Oi)
        {
            #pragma unroll
            for(int Oj=0;Oj<SubOc;++Oj)
                O[(Row+Oi*WarpBlock)*d+OK*Bk+lane+Oj/warpSize] += SubO[Oi*SubOc+Oj+OK*SubOc*SubOr];
        }
    }
}

void CUDA_Kernel::FlashAttention(float*Q,float*K,float*V,float*O,size_t N,size_t d,size_t window)
{
    float *d_q,*d_k,*d_v,*d_o;
    cudaStream_t fa;
    cudaStreamCreate(&fa);
    cudaMallocAsync(&d_q,sizeof(float)*N*d,fa);
    cudaMallocAsync(&d_k,sizeof(float)*d*N,fa);
    cudaMallocAsync(&d_v,sizeof(float)*N*d,fa);
    cudaMallocAsync(&d_o,sizeof(float)*N*d,fa);

    cudaMemcpyAsync(d_q,Q,sizeof(float)*N*d,cudaMemcpyHostToDevice,fa);
    cudaMemcpyAsync(d_k,K,sizeof(float)*N*d,cudaMemcpyHostToDevice,fa);
    cudaMemcpyAsync(d_v,V,sizeof(float)*N*d,cudaMemcpyHostToDevice,fa);
    cudaMemcpyAsync(d_o,O,sizeof(float)*N*d,cudaMemcpyHostToDevice,fa);

    dim3 Block(ThreadPerBlock);
    dim3 grid((N+Br-1)/Br);

    
    if(window*Bc>=N)FlashAttention_16_FullSeq<<<grid,Block>>>(d_q,d_k,d_v,d_o,N,d);
    else SparseWindowAttention<<<grid,Block>>>(d_q,d_k,d_v,d_o,N,d,window);
    cudaStreamSynchronize(fa);
    cudaError_t e = cudaGetLastError();
    if(e!=cudaSuccess)
    {
        const char *estr=cudaGetErrorString(e);
        cudaFree(d_q);cudaFree(d_k);cudaFree(d_v);cudaFree(d_o);
        printf("CUDA ERROR!\n");
        printf("%s\n",estr);
        exit(EXIT_FAILURE); 
    }

    cudaMemcpyAsync(O,d_o,sizeof(float)*N*d,cudaMemcpyDeviceToHost,fa);
    cudaFreeAsync(d_q,fa);
    cudaFreeAsync(d_k,fa);
    cudaFreeAsync(d_v,fa);
    cudaFreeAsync(d_o,fa);
    cudaStreamSynchronize(fa);
}