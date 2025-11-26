#pragma once
#include <cuda.h>
#include <vector>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>  // For at::cuda::philox::unpack
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#ifdef HSTU_CXX_TEST
// hstu c++ test define
std::vector<at::Tensor> hstu_varlen_fwd(
    const at::Tensor& q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor& k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q,  // b+1
    const at::Tensor& cu_seqlens_k,  // b+1
    const int max_seqlen_q,
    const int max_seqlen_k,
    std::optional<const at::Tensor>& num_contexts,  // b
    std::optional<const at::Tensor>& num_targets,  // b
    const int target_group_size,
    int window_size_left,
    int window_size_right,
    const float alpha,
    std::optional<at::Tensor>& rab,
    const bool is_delta_q,
    std::optional<const at::Tensor>& kv_cache,
    std::optional<const at::Tensor>& page_offsets,
    std::optional<const at::Tensor>& page_ids,
    std::optional<const at::Tensor>& last_page_lens,
    std::optional<const at::Tensor>& cu_seqlens_t,
    // extra params
    std::optional<at::Tensor>& attn_mask,
    std::optional<const bool> is_mfalcon,
    std::optional<const at::Tensor>& user_length,
    std::optional<const at::Tensor>& micro_bs,
    std::optional<const at::Tensor>& target_length
    );
#endif