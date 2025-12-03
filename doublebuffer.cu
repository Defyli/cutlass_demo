#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cstdio>   

namespace config{

  using namespace cute;

  template<typename Compute_Type, int KTileM_, int KTileN_, int KTileK_,int NThreads_,int KStage_>
  struct GemeConfig{
    static constexpr int kTileM = KTileM_; 
    static constexpr int kTileN = KTileN_; 
    static constexpr int kTileK = KTileK_; 
    static constexpr int NThreads = NThreads_;
    static_assert(kTileK%8==0);
    static constexpr int PerRowThreads = kTileK/8; // 4 * 8 = 32
    static constexpr int kStage = KStage_;
    
    using T = Compute_Type;
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using TiledMMA = decltype(make_tiled_mma(mma_atom{}, 
                      make_layout(Shape<_2, _2, _1>{}), 
                      make_layout(Shape<_1, _2, _2>{})));

    using SmemALayout = decltype(Layout<
        Shape <Int<kTileM>, Int<kTileK>, Int<kStage>>,
        Stride<Int<kTileK>, _1,          Int<kTileM * kTileK>>>{});

    using SmemBLayout = decltype(Layout<
        Shape <Int<kTileN>, Int<kTileK>, Int<kStage>>,
        Stride<Int<kTileK>, _1,          Int<kTileN * kTileK>>>{});
    
    using Gcopy = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, T>;  
    using GmemTiledCopy = decltype(make_tiled_copy(
          Gcopy{},
          Layout<Shape<Int<NThreads/PerRowThreads>,Int<PerRowThreads>>,Stride<Int<PerRowThreads>,_1>>{},
          Layout<Shape<_1,_8>>{}
    ));
  };

}

template <typename T>
void gen_rand_data(T *data, int n);

template <typename Config>
__global__ void gemm_simple(void *Cptr, const void *Aptr, const void *Bptr, int m, int n, int k) {



  using namespace cute;


  extern __shared__ char smem_buf[];
  using T = typename Config::T;
  T* smem = reinterpret_cast<T*>(smem_buf);
  const int tid = threadIdx.x;

  static_assert(sizeof(T) * Config::kTileK <= 128 ); //确保一个cache line可以放下
  static_assert(Config::kStage == 2); //确保双缓冲


  Tensor sA = make_tensor(make_smem_ptr(smem),typename Config::SmemALayout{});
  Tensor sB = make_tensor(make_smem_ptr(sA.data() + sA.size()),typename Config::SmemBLayout{});


  Tensor A = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(Aptr)), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(reinterpret_cast<const T*>(Bptr)), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(reinterpret_cast<T*>(Cptr)), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor gA = local_tile(A, make_tile(Int<Config::kTileM>{}, Int<Config::kTileK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<Config::kTileN>{}, Int<Config::kTileK>{}), make_coord(ix, _));
  Tensor gC = local_tile(C, make_tile(Int<Config::kTileM>{}, Int<Config::kTileN>{}), make_coord(iy, ix));
  //  gA(kTileM, kTileK, num_tile_k)
  //  gB(kTileN, kTileK, num_tile_k)
  //  gC(kTileM, kTileN) 

  typename Config::TiledMMA tiled_mma;
  typename Config::GmemTiledCopy gmemcopy;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto g2s_copy_ab = gmemcopy.get_slice(tid);

  auto tAgA = g2s_copy_ab.partition_S(gA);
  auto tBgB = g2s_copy_ab.partition_S(gB);
  // (MMA, MMA_M, MMA_K, num_tile_k)
  // (MMA, MMA_N, MMA_K, num_tile_k)

  auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

  auto tAsA = g2s_copy_ab.partition_D(sA); 
  auto tBsB = g2s_copy_ab.partition_D(sB);  

  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)
  Tensor tAsA_mma = thr_mma.partition_A(sA);
  Tensor tBsB_mma = thr_mma.partition_B(sB);
 
  clear(tCrC);
  
  const int num_tile_k = size<2>(gA);
  if(num_tile_k > 1)
  {
    copy(gmemcopy, tAgA(_,_,_,0), tAsA(_,_,_,0));
    copy(gmemcopy, tBgB(_,_,_,0), tBsB(_,_,_,0));
    cute::cp_async_fence(); // 提交异步拷贝
  }

  int write_stage = 1; // 下一次写入 Stage 1
  int read_stage = 0;  // 下一次读取 Stage 0

  for(int itile = 0; itile < num_tile_k - 1; ++itile) {
    // 1. Global -> Shared
    copy(gmemcopy, tAgA(_,_,_,itile+1), tAsA(_,_,_,write_stage));
    copy(gmemcopy, tBgB(_,_,_,itile+1), tBsB(_,_,_,write_stage));
    cute::cp_async_fence();

    
    // 等待当前计算所需的数据 (read_stage) 就绪
    // wait<1> 表示：等待直到只剩下 1 个 batch 在 flight (即刚刚提交的那个 write_stage)
    // 这样 read_stage 就一定加载完成了
    cute::cp_async_wait<1>(); 
    __syncthreads();


    // 3. Shared -> Register
    // 注意：tAsA 是 Copy线程的视图，不能直接 copy 到 MMA线程的寄存器 tArA
    // 需要创建 MMA 视角的 Smem 视图
    
    Tensor tArA = thr_mma.partition_fragment_A(sA(_,_,read_stage)); 
    Tensor tBrB = thr_mma.partition_fragment_B(sB(_,_,read_stage));
    copy(tAsA_mma(_,_,_,read_stage), tArA); 
    copy(tBsB_mma(_,_,_,read_stage), tBrB);

    // 4. Compute
    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);

    __syncthreads();
    
    // 切换 Stage
    write_stage ^= 1;
    read_stage ^= 1;
  }
  // 3. Epilogue: 处理最后一个 Tile
  // 此时不需要再预取了，只需要等待最后一个 Tile 加载完成
  cute::cp_async_wait<0>();
  __syncthreads();

  {
    Tensor tArA = thr_mma.partition_fragment_A(sA(_,_,read_stage)); 
    Tensor tBrB = thr_mma.partition_fragment_B(sB(_,_,read_stage));
    copy(tAsA_mma(_,_,_,read_stage), tArA); 
    copy(tBsB_mma(_,_,_,read_stage), tBrB);
    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  // 8. 将最终结果从寄存器写回全局内存
  cute::copy(tCrC, tCgC);
}

int main() {
  srand(10086);

  using T = cute::half_t;
  using namespace cute;

  T *Cptr;
  T *Aptr;
  T *Bptr;

  int m = 4096;
  int n = 4096;
  int k = 1024;

  cudaMalloc(&Cptr, sizeof(T) * m * n);
  cudaMalloc(&Aptr, sizeof(T) * m * k);
  cudaMalloc(&Bptr, sizeof(T) * k * n);

  T *Aptr_host;
  T *Bptr_host;
  Aptr_host = (T*)malloc(sizeof(T) * m * k);
  Bptr_host = (T*)malloc(sizeof(T) * n * k);
  gen_rand_data(Aptr_host, m * k);
  gen_rand_data(Bptr_host, n * k);

  cudaMemcpy(Aptr, Aptr_host, sizeof(T) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(Bptr, Bptr_host, sizeof(T) * n * k, cudaMemcpyHostToDevice);

  using Config = config::GemeConfig<T, 128, 128, 32, 128, 2>; //double buffer
  Config cf;

  int smem_size = sizeof(T) * (cosize(Config::SmemALayout{}) + cosize(Config::SmemBLayout{}));


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float cute_ms = 0.0f;
  float cublas_ms = 0.0f;


  dim3 block(128);
  dim3 grid(n / cf.kTileN, m / cf.kTileM);
  cudaEventRecord(start);
  for (int i = 0; i < 100; ++i) {
    gemm_simple<decltype(cf)><<<grid, block,smem_size>>>(Cptr, Aptr, Bptr, m, n, k);
  }
  cudaEventRecord(stop);
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&cute_ms, start, stop);

  auto err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  // cublas
  T *Cptr_cublas;

  cudaMalloc(&Cptr_cublas, sizeof(T) * m * n);

  cublasHandle_t handle;
  cublasCreate(&handle);

  half alpha = half(1.f);
  half beta = half(0.f);

  cudaEventRecord(start);
  for (int i = 0; i < 100; ++i) {
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
          	  n, m, k,
          	  &alpha,
          	  (half *)Bptr, k,
          	  (half *)Aptr, k,
          	  &beta,
          	  (half *)Cptr_cublas, n);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      printf("blas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }
  }
  cudaEventRecord(stop);
  cudaDeviceSynchronize();

  cudaEventElapsedTime(&cublas_ms, start, stop);



  err = cudaGetLastError();
  printf("err = %d, str = %s\n", err, cudaGetErrorString(err));

  T *Cptr_host;
  T *Cptr_cublas_host;

  Cptr_host = (T*)malloc(sizeof(T) * m * n);
  Cptr_cublas_host = (T*)malloc(sizeof(T) * m * n);

  // compare
  cudaMemcpy(Cptr_host, Cptr, sizeof(T) * m * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(Cptr_cublas_host, Cptr_cublas, sizeof(T) * m * n, cudaMemcpyDeviceToHost);

  float threshold = 0.1;
  for (int i = 0; i < m * n; ++i) {
    float v1 = Cptr_host[i];
    float v2 = Cptr_cublas_host[i];
    if (fabs(v2 - v1) > threshold) {
      printf("v1 = %f, v2 = %f\n", v1, v2);
    }
  }

  Tensor tensor_C = make_tensor(Cptr_host, make_shape(m, n), make_stride(n, 1));
  Tensor tensor_C_cublas = make_tensor(Cptr_cublas_host, make_shape(m, n), make_stride(n, 1));

  auto tile = make_tile(8, 8);
  auto coor = make_coord(0, 0);
  Tensor tc1 = local_tile(tensor_C, tile, coor);
  Tensor tc1_cublas = local_tile(tensor_C_cublas, tile, coor);

  print_tensor(tc1);
  print_tensor(tc1_cublas);



  double total_flops = 2.0 * m * n * k;
  printf("\n--- Performance ---\n");
  printf("CUTE   Kernel: Total time: %.3f ms, Avg time: %.3f ms, TFLOPS: %.2f\n",
         cute_ms, cute_ms / 100.0, (total_flops * 100 / (cute_ms / 1000.0)) / 1e12);
  printf("cuBLAS Kernel: Total time: %.3f ms, Avg time: %.3f ms, TFLOPS: %.2f\n",
         cublas_ms, cublas_ms / 100.0, (total_flops * 100 / (cublas_ms / 1000.0)) / 1e12);
  printf("-------------------\n\n");



  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

template <typename T>
void gen_rand_data(T *data, int n) {
  for (int i = 0; i < n; ++i) {
    float v = (rand() % 200 - 100) * 0.01;
    data[i] = v;
  }
}