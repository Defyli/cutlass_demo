#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "hstu.h"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>
#include "block_info.h"
#include "kernel_traits.h"
#include "static_switch.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <typename Kernel_traits, typename Params>
inline __device__ void hstu_compute_attn_1rowblock_mask(const Params& params,
                                                        const int bidb,
                                                        const int bidh,
                                                        int m_block) {
    // This function is a copy of hstu_compute_attn_1rowblock from hstu_fwd.h,
    // with the addition of dynamic mask handling.

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    constexpr bool Has_rab = Kernel_traits::Has_rab;
    constexpr bool Has_mask = Kernel_traits::Has_mask; // This will be true for this kernel
    static_assert(Has_mask, "hstu_compute_attn_1rowblock_mask is only for kernels with a mask.");

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr bool Is_even_rab = Kernel_traits::Is_even_rab;
    constexpr int UserLen = Kernel_traits::User_len;

    const HstuBlockInfo<Kernel_traits, Params> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) {
        return;
    }

    // Standard HSTU setup (copied from hstu_fwd.h)
    const int actual_seqlen_q = binfo.actual_seqlen_q;
    const int actual_seqlen_k = binfo.actual_seqlen_k;
    const int n_block_max = cute::ceil_div(actual_seqlen_k, kBlockN);
    const int n_block_min = 0;

    // Global memory tensors (Q, K, V, O)
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_row_stride)),
                            make_shape(actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + binfo.k_offset(params.k_row_stride)),
                            make_shape(actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + binfo.k_offset(params.v_row_stride)),
                            make_shape(actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));


    const int bidh_rab = (params.h_rab > 1) ? bidh : 0;
    size_t rab_qkv_not_equal_offset = bidb * params.rab_seqlen_qk_stride + bidh_rab * params.rab_seqlen_q_stride;

    //  Global memory tensor for the dynamic mask
    Tensor mMask = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.mask_ptr) + bidb * params.mask_batch_stride),
                               make_shape(actual_seqlen_q, params.seqlen_k_rounded),
                               make_stride(params.mask_row_stride, _1{}));
    Tensor gMask = local_tile(mMask, Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block, _));
    Tensor mRab =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.rab_ptr) + rab_qkv_not_equal_offset),
                  make_shape(actual_seqlen_q, params.seqlen_k_rounded),
                  make_stride(params.rab_seqlen_k_stride, _1{}));
    Tensor gRab = local_tile(mRab(_, _),
                        make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                        make_coord(m_block, _));


    // Shared memory setup (including sMask)
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    Tensor sRab = make_tensor(sV.data() + size(sV), typename Kernel_traits::SmemLayoutRab{});
    Tensor sMask = make_tensor(sRab.data() + (Kernel_traits::Has_rab ? size(sRab) : 0), typename Kernel_traits::SmemLayoutMask{});

    // Tiled copies (including for mask)
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyMask gmem_tiled_copy_Mask;
    auto gmem_thr_copy_Mask = gmem_tiled_copy_Mask.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyRab gmem_tiled_copy_Rab;
    auto gmem_thr_copy_Rab = gmem_tiled_copy_Rab.get_thread_slice(tidx);

    // Partition tensors (including for mask)
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tQgMask = gmem_thr_copy_Mask.partition_S(gMask);
    Tensor tQsMask = gmem_thr_copy_Mask.partition_D(sMask);
    Tensor tQgRab = gmem_thr_copy_Rab.partition_S(gRab);
    Tensor tQsRab = gmem_thr_copy_Rab.partition_D(sRab);

    // MMA setup
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK(_, _, _0{}));
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle(_, _, _0{}));
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    auto smem_tiled_copy_rab = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_rab = smem_tiled_copy_rab.get_thread_slice(tidx);
    Tensor tSsRab = smem_thr_copy_rab.partition_S(sRab);
    auto smem_tiled_copy_mask = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_mask = smem_tiled_copy_mask.get_thread_slice(tidx);
    Tensor tSsMask = smem_thr_copy_mask.partition_S(sMask);

    // Predicates
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
    auto cRab = make_identity_tensor(make_shape(size<0>(sRab), size<1>(sRab)));
    auto tQcRab = gmem_thr_copy_Rab.partition_S(cRab);
    auto cMask = make_identity_tensor(make_shape(size<0>(sMask), size<1>(sMask)));
    auto tQcMask = gmem_thr_copy_Mask.partition_S(cMask);


    auto copy_if_g2s_rab = [&](int n_block_id, int buffer_stage) {
        auto ctQgRab_view = tQgRab(_, _, _, n_block_id);
        #pragma unroll
        for (int m = 0; m < size<1>(ctQgRab_view); ++m) {
        if (get<0>(tQcRab(0, m, 0)) < (actual_seqlen_q - m_block * kBlockM)) {
            #pragma unroll
            for (int k = 0; k < size<2>(ctQgRab_view); ++k) {
                if (Is_even_rab || get<1>(tQcRab(0, m, k)) < (actual_seqlen_k - n_block_id * kBlockN)) {
                    cute::copy(gmem_tiled_copy_Rab, ctQgRab_view(_, m, k), tQsRab(_, m, k, buffer_stage));
                }
            }
        }
        }
    };

    auto copy_g2s_rab = [&](int n_block_id, int buffer_stage) {
        auto ctQgRab_view = tQgRab(_, _, _, n_block_id);
        #pragma unroll
        for (int m = 0; m < size<1>(ctQgRab_view); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(ctQgRab_view); ++k) {
            if (Is_even_rab || get<0>(tQcRab(0, m, k)) < (actual_seqlen_q - m_block * kBlockM)) {
            cute::copy(gmem_tiled_copy_Rab, ctQgRab_view(_, m, k), tQsRab(_, m, k, buffer_stage));
            }
        }
        }
    };

    auto copy_g2s_mask = [&](int n_block_id, int buffer_stage) {
        auto ctQgMask_view = tQgMask(_, _, _, n_block_id);
        #pragma unroll
        for(int m=0;m<size<1>(ctQgMask_view);++m)
        {
            if(get<0>(tQcMask(0,m,0))<(actual_seqlen_q-m_block*kBlockM))
            {
                #pragma unroll
                for(int k=0;k<size<2>(ctQgMask_view);++k)
                {
                    cute::copy(gmem_tiled_copy_Mask,ctQgMask_view(_,m,k),tQsMask(_,m,k,buffer_stage));
                }
            }
        }
    };


    int buffer_stage = 0;
    int n_block = 0;
    //since rab is ailgened to 16 bytes, no need to do dual check
    if constexpr(Has_rab){
        copy_if_g2s_rab(n_block,buffer_stage);
    }

    // Prologue
    flash::copy</*Is_even_MN=*/false>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, actual_seqlen_q - m_block*kBlockM);
    if constexpr (Kernel_traits::Is_Q_in_regs) {
        cute::cp_async_fence();
    }

    if constexpr (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }


    auto tKsK_stage_view = tKsK(_, _, _, buffer_stage);
    flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/false>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK_stage_view, tKVcKV, actual_seqlen_k - n_block * kBlockN);
    
    // New: Load first mask tile
    copy_g2s_mask(n_block,buffer_stage);

    cute::cp_async_fence();
    if constexpr (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }
    clear(acc_o);

    // Main loop
    for (; n_block < n_block_max; ++n_block) {
        if constexpr(UserLen>0)
        {
            // user part
            bool is_q_user_block = (m_block + 1) * kBlockM <= UserLen;
            // target part
            bool is_k_target_block = (n_block * kBlockN) >= UserLen;

            if (is_q_user_block && is_k_target_block) {
                // skip
                break;
            }
        }

        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // Load V tile
        auto tVsV_stage_view = tVsV(_, _, _, buffer_stage);
        flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV_stage_view, tKVcKV, actual_seqlen_k - n_block * kBlockN);
        cute::cp_async_fence();

        if constexpr(Has_rab)
        {
            Tensor rRab = make_tensor<Element>(
                partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
            auto tSrRab_view = smem_thr_copy_rab.retile_D(rRab);
            cute::copy(smem_tiled_copy_rab, tSsRab(_, _, _, buffer_stage), tSrRab_view(_, _, _));
            flash::convert_type_safe(rRab, acc_s);
            //copy next rab
            auto tQsRab_view = tQsRab(_,_,_,buffer_stage);
            clear(tQsRab_view);
            if(n_block<n_block_max){
                copy_g2s_rab(n_block+1,buffer_stage);
            }
        }
        else{
            clear(acc_s);
        }

        // Compute Q @ K
        flash::gemm<Kernel_traits::Is_Q_in_regs>(acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, buffer_stage), tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        // New: Apply the mask
        Tensor rMask = make_tensor<Element>(partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
        auto tSrMask_view = smem_thr_copy_mask.retile_D(rMask);
        cute::copy(smem_tiled_copy_mask, tSsMask(_, _, _, buffer_stage), tSrMask_view);
        #pragma unroll
        for(int i = 0; i < size(acc_s); ++i) {
            acc_s(i) *= static_cast<ElementAccum>(rMask(i));
        }

        flash::cp_async_wait<0>();
        __syncthreads();

        // Load next K and Mask tiles
        if (n_block <n_block_max) {
            auto tKsK_stage_view_next = tKsK(_, _, _, buffer_stage);
            flash::copy</*Is_even_MN=*/true>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block + 1), tKsK_stage_view_next, tKVcKV);
            auto tQsMask_view_next = tQsMask(_, _, _, buffer_stage);
            clear(tQsMask_view_next);
            copy_g2s_mask(n_block+1,buffer_stage);     
            cute::cp_async_fence();
        }

        // Scale, silu, and compute P @ V

        for (int i = 0; i < size(acc_s); ++i) { acc_s(i) *= params.alpha; }
        fast_silu(acc_s);
        
        Tensor rP = make_tensor_like<Element>(acc_s);
        flash::convert_type_safe(acc_s, rP);
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt(_, _, _, buffer_stage), tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

 // scale acc_o
  for (int i = 0; i < size(acc_o); ++i) {
    acc_o(i) /= params.seqlen_q;
  }

  // Epilogue
  // Convert acc_o from fp32 to fp16/bf16
  Tensor rO = make_tensor_like<Element>(acc_o);
  flash::convert_type_safe(acc_o, rO);
  Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
  // Partition sO to match the accumulator partitioning
  auto smem_tiled_copy_O =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

  if constexpr (Kernel_traits::Share_Q_K_smem) {
    __syncthreads();
  }

  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  Tensor mO =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) +
                                binfo.q_offset(params.o_row_stride)),
                  make_shape(actual_seqlen_q, params.h, params.d),
                  make_stride(params.o_row_stride, params.o_head_stride, _1{}));
  Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));

  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  __syncthreads();

  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  // Construct identity layout for sO
  Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
  // Repeat the partitioning with identity layouts
  Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
  Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
  // Clear_OOB_K must be false since we don't want to write zeros to gmem
  flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
      gmem_tiled_copy_O, tOrO, tOgO, tOcO, actual_seqlen_q - m_block * kBlockM);
}

// Define the __global__ kernel that calls the device function
template <typename Kernel_traits, typename Params>
__global__ void hstu_fwd_kernel_mask(Params params) {
    // Same grid calculation as hstu_fwd_kernel
    int m_block, bidb, bidh;
    m_block = gridDim.x - blockIdx.x - 1;
    bidh = blockIdx.y;
    bidb = blockIdx.z;
    hstu_compute_attn_1rowblock_mask<Kernel_traits>(params, bidb, bidh, m_block);
}

} // namespace flash