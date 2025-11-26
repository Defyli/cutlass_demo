import torch
from hstu_attn import hstu_attn_varlen_func

def run_hstu(
        q:torch.Tensor(),
        k:torch.Tensor(),
        v:torch.Tensor(),
        cu_seqlens_q:torch.Tensor(),
        cu_seqlens_k:torch.Tensor(),
        max_seqlen_q:int,
        max_seqlen_k:int,
        user_length:torch.Tensor(),
        micro_bs:torch.Tensor(),
        target_length:torch.Tensor(),
        mask:torch.Tensor() = None
)->torch.Tensor():
    res=hstu_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        None,None,
        1,-1,-1,1,
        None,False,None,None,None,None,None,mask,
        True,user_length,micro_bs,target_length
    )
    return res[0]


def run_mask(
        q:torch.Tensor(),
        k:torch.Tensor(),
        v:torch.Tensor(),
        cu_seqlens_q:torch.Tensor(),
        cu_seqlens_k:torch.Tensor(),
        max_seqlen_q:int,
        max_seqlen_k:int,
        mask:torch.Tensor() = None
)->torch.Tensor():
    res=hstu_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        None,None,
        1,-1,-1,1,
        None,False,None,None,None,None,None,mask,False,None,None,None
    )
    return res[0]


def run_mask_rab(q:torch.Tensor(),
        k:torch.Tensor(),
        v:torch.Tensor(),
        cu_seqlens_q:torch.Tensor(),
        cu_seqlens_k:torch.Tensor(),
        max_seqlen_q:int,
        max_seqlen_k:int,
        mask:torch.Tensor() = None,
        rab:torch.Tensor() = None):
    res=hstu_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        None,None,
        1,-1,-1,1,
        rab,False,None,None,None,None,None,mask,False,None,None,None
    )
    return res[0]