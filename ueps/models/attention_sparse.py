from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.integrations.flex_attention import compile_friendly_flex_attention
from torch.nn.attention.flex_attention import BlockMask

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def sdpa_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Arguments:
        q: (batch_size, nheads, seqlen, headdim)
        k: (batch_size, nheads_k, seqlen, headdim)
        v: (batch_size, nheads_k, seqlen, headdim)
    Return:
        out: (batch_size, seqlen, nheads, headdim)
    """
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
    )
    return attn_output.transpose(1, 2).contiguous()

def flex_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, BlockMask, None],
    module: nn.Module,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    data_type: Optional[torch.dtype] = torch.bfloat16,
    **kwargs,
) -> torch.Tensor:
    """
    Arguments:
        q: (batch_size, nheads, seqlen, headdim)
        k: (batch_size, nheads_k, seqlen, headdim)
        v: (batch_size, nheads_k, seqlen, headdim)
    Return:
        out: (batch_size, seqlen, nheads, headdim)
    """
    if kwargs.get("dropout", 0.0) > 0:
        raise ValueError(
            "`flex_attention` does not support `dropout`. Please use it with inference only (model.eval()) "
            "or disable attention dropout in the config."
        )

    # Cast to a lower precision when running in float32 for speed
    target_dtype: Optional[torch.dtype] = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        else:
            target_dtype = data_type
    if target_dtype is not None and query.dtype == torch.float32:
        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    block_mask = None
    score_mask = None
    if isinstance(attention_mask, BlockMask):
        block_mask = attention_mask
    else:
        score_mask = attention_mask

    if score_mask is not None:
        # Ensure mask aligns with kv length
        score_mask = score_mask[:, :, :, : key.shape[-2]]

    def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
        if softcap is not None:
            score = softcap * torch.tanh(score / softcap)
        if score_mask is not None:
            score = score + score_mask[batch_idx][0][q_idx][kv_idx]
        return score

    flex_attention_output = compile_friendly_flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scaling,
        training=module.training,
    )

    attention_output = flex_attention_output.transpose(1, 2).contiguous()
    return attention_output

def math_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    module: Optional[nn.Module] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Manual attention implementation that stores attention weights in the module.
    """
    import math
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.size(-1))
    
    print(f"query shape: {query.shape}")
    
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    
    if dropout > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout)
        
    if module is not None:
        module.attn_weights = attn_weights
        
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous()


SUPPORTED_ATTENTION_IMPLS = (
    "sdpa",
    "flex_attention",
    "math",
)
ATTENTION_IMPLS = {
    "sdpa": sdpa_attention_forward,
    "flex_attention": flex_attention_forward,
    "math": math_attention_forward,
}
