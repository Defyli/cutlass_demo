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

namespace flash{
using namespace cute;
template <typename Kernel_traits, typename Params>
inline __device__ void hstu_compute_attn_1rowblock_mfalcon(const Params& params,
                                                        const int bidb,
                                                        const int bidh,
                                                        int m_block_global) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    constexpr bool Has_rab = Kernel_traits::Has_rab;
    constexpr bool Has_mask = Kernel_traits::Has_mask;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr float mask_value = -std::numeric_limits<float>::infinity();

    // M-Falcon parameters
    const int user_length = params.user_length[bidb];
    const int target_length = params.target_length[bidb];
    const int micro_bs = params.micro_bs[bidb];
    const int total_q_len = user_length + target_length;

    // Early exit for blocks outside the valid query length
    if (m_block_global * kBlockM >= total_q_len) {
        return;
    }

    // Determine the current step and effective sequence lengths for this m_block
    const int m_block_q_start_row = m_block_global * kBlockM;
    
    int k_len_prompt, k_len_target;
    int k_target_start_row_in_k; // The global row index in the original K tensor where the target part for this step begins.

    if (m_block_q_start_row < user_length) { // Query is in the user prompt part -> "Prefill" step
        k_len_prompt = user_length;
        k_len_target = micro_bs;
        k_target_start_row_in_k = user_length;
    } else { // Query is in the target part -> "Loop" step
        int step = (m_block_q_start_row - user_length) / micro_bs;
        k_len_prompt = user_length;
        k_len_target = micro_bs;
        k_target_start_row_in_k = user_length + step * micro_bs;
    }
    const int k_len_effective = k_len_prompt + k_len_target;

    // calculate n_block_min and n_block_max
    const int n_block_min = 0;
    int n_block_max = cute::ceil_div(k_len_effective, kBlockN);
    
    // Base pointers for the current batch item
    const int sum_s_q = params.cu_seqlens_q[bidb];
    const int sum_s_k = params.cu_seqlens_k[bidb];
    Element* q_base_ptr = reinterpret_cast<Element*>(params.q_ptr) + sum_s_q * params.q_row_stride;
    Element* k_base_ptr = reinterpret_cast<Element*>(params.k_ptr) + sum_s_k * params.k_row_stride;
    Element* v_base_ptr = reinterpret_cast<Element*>(params.v_ptr) + sum_s_k * params.v_row_stride ;
    Element* o_base_ptr = reinterpret_cast<Element*>(params.o_ptr) + sum_s_q * params.o_row_stride;
    const int bidh_rab = (params.h_rab > 1) ? bidh : 0;
    Element* rab_base_ptr = reinterpret_cast<Element*>(params.rab_ptr) + bidb * params.rab_seqlen_qk_stride + bidh_rab * params.rab_seqlen_q_stride;
    Element* mask_base_ptr = reinterpret_cast<Element*>(params.mask_ptr) + bidb * params.mask_batch_stride;
    // We exit early if this block of Q has no K to attend to.
    if (n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(o_base_ptr)),
                                make_shape(total_q_len, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block_global, 0));
        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, total_q_len - m_block_global * kBlockM);
        return;
    }

    // Tensors for Q and O for this m_block
    Tensor mQ_total = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(q_base_ptr)),
                            make_shape(total_q_len, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ_total(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block_global, 0));

    // Tensors for the different parts of K and V
    Tensor mK_prompt = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(k_base_ptr)),
                                make_shape(user_length, params.h_k, params.d),
                                make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor mV_prompt = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(v_base_ptr)),
                                make_shape(user_length, params.h_k, params.d),
                                make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    
    Tensor mK_target = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(k_base_ptr) + k_target_start_row_in_k * params.k_row_stride),
                                make_shape(micro_bs, params.h_k, params.d),
                                make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor mV_target = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(v_base_ptr) + k_target_start_row_in_k * params.v_row_stride),
                                make_shape(micro_bs, params.h_k, params.d),
                                make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    

    Tensor mRab_prompt = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(rab_base_ptr)),
                                     make_shape(total_q_len, user_length),
                                     make_stride(params.rab_seqlen_k_stride, _1{}));

    Tensor mRab_target = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(rab_base_ptr) + k_target_start_row_in_k),  // row offset
                                     make_shape(total_q_len, micro_bs),
                                     make_stride(params.rab_seqlen_k_stride, _1{}));
    
    Tensor mMask_prompt = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(mask_base_ptr)),
                                     make_shape(total_q_len, user_length),
                                     make_stride(params.mask_row_stride, _1{}));
    Tensor mMask_target = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(mask_base_ptr) + k_target_start_row_in_k),  // column offset
                                     make_shape(total_q_len, micro_bs),
                                     make_stride(params.mask_row_stride, _1{}));
    

    // Shared memory setup
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    Tensor sRab = make_tensor(sV.data() + size(sV), typename Kernel_traits::SmemLayoutRab{});
    Tensor sMask = make_tensor(sRab.data() + (Kernel_traits::Has_rab?size(sRab):0), typename Kernel_traits::SmemLayoutMask{});
    // Tiled copies
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyRab gmem_tiled_copy_Rab;
    auto gmem_thr_copy_Rab = gmem_tiled_copy_Rab.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyMask gmem_tiled_copy_Mask;
    auto gmem_thr_copy_Mask = gmem_tiled_copy_Mask.get_thread_slice(tidx);

    // Partition source and destination tensors
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

    Tensor gK_prompt_tile = local_tile(mK_prompt(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));
    Tensor gK_target_tile = local_tile(mK_target(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));
    Tensor gV_prompt_tile = local_tile(mV_prompt(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));
    Tensor gV_target_tile = local_tile(mV_target(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));
    Tensor gRab_prompt_tile = local_tile(mRab_prompt, Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block_global, _));
    Tensor gRab_target_tile = local_tile(mRab_target, Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block_global, _));

    Tensor gMask_prompt_tile = local_tile(mMask_prompt, Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block_global, _));
    Tensor gMask_target_tile = local_tile(mMask_target, Shape<Int<kBlockM>, Int<kBlockN>>{}, make_coord(m_block_global, _));

    Tensor tKgK_prompt = gmem_thr_copy_QKV.partition_S(gK_prompt_tile);
    Tensor tKgK_target = gmem_thr_copy_QKV.partition_S(gK_target_tile);
    Tensor tVgV_prompt = gmem_thr_copy_QKV.partition_S(gV_prompt_tile);
    Tensor tVgV_target = gmem_thr_copy_QKV.partition_S(gV_target_tile);
    Tensor tQgRab_prompt = gmem_thr_copy_Rab.partition_S(gRab_prompt_tile);
    Tensor tQgRab_target = gmem_thr_copy_Rab.partition_S(gRab_target_tile);
    Tensor tQsRab = gmem_thr_copy_Rab.partition_D(sRab);

    Tensor tQgMask_prompt = gmem_thr_copy_Mask.partition_S(gMask_prompt_tile);
    Tensor tQgMask_target = gmem_thr_copy_Mask.partition_S(gMask_target_tile);
    Tensor tQsMask = gmem_thr_copy_Mask.partition_D(sMask);

    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    

    // MMA setup
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK(_, _, _0{}));
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle(_, _, _0{}));
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    // Smem copy setup
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
    auto tSsRab = smem_thr_copy_rab.partition_S(sRab);

    auto smem_tiled_copy_mask = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_mask = smem_tiled_copy_mask.get_thread_slice(tidx);
    auto tSsMask = smem_thr_copy_mask.partition_S(sMask);

    // Predicates
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
    auto cRab = make_identity_tensor(make_shape(size<0>(sRab), size<1>(sRab)));
    auto tQcRab = gmem_thr_copy_Rab.partition_S(cRab);
    auto cMask = make_identity_tensor(make_shape(size<0>(sMask), size<1>(sMask)));
    auto tQcMask = gmem_thr_copy_Mask.partition_S(cMask);


    // Prologue: load Q
    flash::copy</*Is_even_MN=*/false>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, total_q_len - m_block_q_start_row);
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

    int buffer_stage = 0;

    /// Prologue: Load first K, V, RAB, MASK
    int n_block = n_block_max - 1;
    const int k_col_offset_prologue = n_block * kBlockN;
    auto tKsK_stage_view = tKsK(_, _, _, buffer_stage);
    auto tVsV_stage_view = tVsV(_, _, _, buffer_stage);
    auto tQsRab_view = tQsRab(_, _, _, buffer_stage);
    auto tQsMask_view = tQsMask(_, _, _, buffer_stage);
    if (k_col_offset_prologue < k_len_prompt) {
        flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/false>(
            gmem_tiled_copy_QKV, tKgK_prompt(_, _, _, n_block), tKsK_stage_view, tKVcKV,
            k_len_prompt - k_col_offset_prologue);
        flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV_prompt(_, _, _, n_block), tVsV_stage_view, tKVcKV, 
            k_len_prompt - k_col_offset_prologue);
        if constexpr (Has_rab) {
            clear(tQsRab_view);
            flash::copy_with_dual_bound_check(
                gmem_tiled_copy_Rab, tQgRab_prompt(_, _, _, n_block), tQsRab_view, tQcRab,
                total_q_len - m_block_q_start_row, k_len_prompt - k_col_offset_prologue);
        }
        if constexpr (Has_mask) {
            clear(tQsMask_view);
            flash::copy_with_dual_bound_check(
                gmem_tiled_copy_Mask, tQgMask_prompt(_, _, _, n_block), tQsMask_view, tQcMask,
                total_q_len - m_block_q_start_row, k_len_prompt - k_col_offset_prologue);
        }
    } else {
        int target_n_block = (k_col_offset_prologue - k_len_prompt) / kBlockN;
        flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/false>(
            gmem_tiled_copy_QKV, tKgK_target(_, _, _, target_n_block), tKsK_stage_view, tKVcKV,
            k_len_effective - k_col_offset_prologue);
        flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV_target(_, _, _, target_n_block), tVsV_stage_view, tKVcKV, 
            k_len_effective - k_col_offset_prologue);
        if constexpr (Has_rab) {
            clear(tQsRab_view);
            flash::copy_with_dual_bound_check(
                gmem_tiled_copy_Rab, tQgRab_target(_, _, _, target_n_block), tQsRab_view, tQcRab,
                total_q_len - m_block_q_start_row, k_len_effective - k_col_offset_prologue);
        }
        if constexpr (Has_mask) {
            clear(tQsMask_view);
            flash::copy_with_dual_bound_check(
                gmem_tiled_copy_Mask, tQgMask_target(_, _, _, target_n_block), tQsMask_view, tQcMask,
                total_q_len - m_block_q_start_row, k_len_effective - k_col_offset_prologue);
        }
    }
    cute::cp_async_fence();

    if constexpr (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    auto apply_mask = [&](auto& tensor, int n_block_k) {
        static constexpr int Row = 0, Col = 1;
        Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thr_mma.partition_C(cS);
        
        #pragma unroll
        for (int i = 0; i < size(tensor); ++i) {
            const int row_in_block = get<Row>(tScS(i));
            const int col_in_block = get<Col>(tScS(i));
            
            const int row_global = m_block_q_start_row + row_in_block;
            const int col_k_cache = n_block_k * kBlockN + col_in_block;

            if (col_k_cache >= k_len_effective) {
                tensor(i) = mask_value;
                continue;
            }

            int col_global;
            if (col_k_cache < user_length) { // K is in prompt part
                col_global = col_k_cache;
            } else { // K is in target part
                col_global = k_target_start_row_in_k + (col_k_cache - user_length);
            }

            // Rule: user query cannot attend to target key.
            if (row_global < user_length && col_global >= user_length) {
                tensor(i) = mask_value;
                continue;
            }

            // M-Falocn Rule: target query can only attend to formers in target keys (diagonal attention).
            if (row_global >= user_length && col_global >= user_length) {
                if (row_global >= col_global) {
                    tensor(i) = mask_value;
                    continue;
                }
            }
        }
    };

    

    // Main loop over K blocks
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        const int k_col_offset_next = (n_block - 1) * kBlockN;
        flash::cp_async_wait<0>();
        __syncthreads();

        // Load V
        auto tVsV_stage_view = tVsV(_, _, _, buffer_stage);
        const int v_col_offset = n_block * kBlockN;
        if (v_col_offset < k_len_prompt) {
            flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV_prompt(_, _, _, n_block), tVsV_stage_view, tKVcKV, k_len_prompt - v_col_offset);
        } else {
            int target_n_block = (v_col_offset - k_len_prompt) / kBlockN;
            flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV_target(_, _, _, target_n_block), tVsV_stage_view, tKVcKV, k_len_effective - v_col_offset);
        }
        cute::cp_async_fence();

        if constexpr (Has_rab) {
            Tensor rRab = make_tensor<Element>(partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
            auto tSrRab_view = smem_thr_copy_rab.retile_D(rRab);
            cute::copy(smem_tiled_copy_rab, tSsRab(_, _, _, buffer_stage), tSrRab_view);
            flash::convert_type_safe(rRab, acc_s);
            //copy next rab
            auto tQsRab_view = tQsRab(_,_,_,buffer_stage);
            clear(tQsRab_view);
            if(n_block>n_block_min){
                if (k_col_offset_next < k_len_prompt) {
                    flash::copy_with_dual_bound_check(gmem_tiled_copy_Rab, 
                        tQgRab_prompt(_, _, _, n_block - 1), tQsRab_view , 
                        tQcRab, kBlockM, kBlockN);
                } else {
                    int target_n_block = (k_col_offset_next - k_len_prompt) / kBlockN;
                    flash::copy_with_dual_bound_check(gmem_tiled_copy_Rab, 
                        tQgRab_target(_, _, _, target_n_block), tQsRab_view ,
                        tQcRab, kBlockM, kBlockN);
                }
            }
        } else {
            clear(acc_s);
        }

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, buffer_stage), tiled_mma, smem_tiled_copy_Q,
            smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

        if constexpr (Has_mask) {
            Tensor rMask = make_tensor<Element>(partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
            auto tSrMask_view = smem_thr_copy_mask.retile_D(rMask);
            cute::copy(smem_tiled_copy_mask, tSsMask(_, _, _, buffer_stage), tSrMask_view);
            #pragma unroll
            for(int i=0;i<size(acc_s);++i)
                acc_s(i) *= static_cast<ElementAccum>(rMask(i));
        }
        else{
            apply_mask(acc_s, n_block);
        }
        flash::cp_async_wait<0>();
        __syncthreads();

        // Load next K block and mask block
        if (n_block > n_block_min) {
            auto tKsK_stage_view_next = tKsK(_, _, _, buffer_stage);
            auto tQsMask_view_next = tQsMask(_,_,_,buffer_stage);
            if (k_col_offset_next < k_len_prompt) {
                flash::copy</*Is_even_MN=*/true>(gmem_tiled_copy_QKV,
                                                tKgK_prompt(_, _, _, n_block - 1),
                                                tKsK_stage_view_next, tKVcKV);
                if constexpr(Has_mask)
                {
                    clear(tQsMask_view_next);
                    flash::copy_with_dual_bound_check(gmem_tiled_copy_Mask, 
                        tQgMask_prompt(_, _, _, n_block - 1), tQsMask_view_next, 
                        tQcMask, total_q_len - m_block_q_start_row, k_len_prompt - k_col_offset_next);
                }
            } else {
                int target_n_block = (k_col_offset_next - k_len_prompt) / kBlockN;
                flash::copy</*Is_even_MN=*/true>(gmem_tiled_copy_QKV,
                                                tKgK_target(_, _, _, target_n_block),
                                                tKsK_stage_view_next, tKVcKV);
                if constexpr(Has_mask)
                {
                    clear(tQsMask_view_next);
                    flash::copy_with_dual_bound_check(gmem_tiled_copy_Mask, 
                        tQgMask_target(_, _, _, target_n_block), tQsMask_view_next, 
                        tQcMask, total_q_len - m_block_q_start_row, k_len_effective - k_col_offset_next);
                }
            }

            cute::cp_async_fence();
        }

        for (int i = 0; i < size(acc_s); ++i) {
            acc_s(i) *= params.alpha;
        }
        fast_silu(acc_s);

        Tensor rP = make_tensor_like<Element>(acc_s);
        flash::convert_type_safe(acc_s, rP);

        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));

        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt(_, _, _, buffer_stage), tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue
    for (int i = 0; i < size(acc_o); ++i) {
        acc_o(i) /= params.seqlen_q; // Normalize by total length
    }

    Tensor rO = make_tensor_like<Element>(acc_o);
    flash::convert_type_safe(acc_o, rO);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    if constexpr (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
    }
 
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(o_base_ptr)),
                            make_shape(total_q_len, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block_global, 0));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, total_q_len - m_block_global * kBlockM);
}




template <typename Kernel_traits, typename Params>
__global__ void hstu_mfalcon_persistent(Params params)
{
  constexpr bool Is_balance = Kernel_traits::Is_balance;
  int m_block, bidb, bidh;
  if constexpr (Is_balance) {
    m_block = gridDim.z - blockIdx.z - 1;
    bidh = blockIdx.x;
    bidb = blockIdx.y;
  } else {
    m_block = gridDim.x - blockIdx.x - 1;
    bidh = blockIdx.y;
    bidb = blockIdx.z;
  }

  hstu_compute_attn_1rowblock_mfalcon<Kernel_traits>(params,bidb,bidh,m_block);
 }      
}

