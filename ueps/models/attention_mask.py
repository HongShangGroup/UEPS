from collections.abc import Callable
from typing import Optional, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

from torch.nn.attention.flex_attention import BlockMask, create_block_mask
_DEFAULT_SPARSE_BLOCK_SIZE = 128

def _vmap_for_bhqkv(mask_function: Callable, bh_indices: bool = True) -> Callable:
    """
    Used to vmap our mask_functions over the q_idx and kv_idx dimensions of the inputs. Optionally, vmap over
    the batch and head indices as well if `bh_indices=True`.
    Using vmap here allows us to keep the performance of vectorized ops, while having a single set of primitive
    functions between attention interfaces (i.e. between flex and sdpa/eager, FA2 being a bit different).

    Args:
        mask_function (`Callable`):
            The mask_function to vmap.
        bh_indices (`bool`, optional):
            Whether to vmap over the batch and head indices as well, or only q and kv indices.

    Returns:
        Callable: The vmapped function.
    """
    # We vmap the function 2 times, broadcasting the [q_idx, kv_idx] dimensions
    dimensions = [(None, None, None, 0), (None, None, 0, None)]
    if bh_indices:
        # We extend broadcasting over the [batch_idx, head_idx] dimensions
        dimensions.extend([(None, 0, None, None), (0, None, None, None)])

    for dims in dimensions:
        mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
    return mask_function

def sliding_chunked_mask_function(chunk_size: int, sliding_chunk: int) -> Callable:
    """Token-level mask function for flex's create_block_mask: allow only within same chunk."""
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        q_chunk = q_idx // chunk_size
        kv_chunk = kv_idx // chunk_size
        return abs(q_chunk - kv_chunk) <= sliding_chunk
    return inner_mask

def sdpa_mask(
    q_length: int,
    kv_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    mask_function: Callable = None,
    device: Optional[torch.device] = "cuda",
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
    the element should take part in the attention computation, and False that it should not.
    """
    kv_length = kv_length if kv_length is not None else q_length

    q_range = torch.arange(q_length, device=device)
    kv_arange = torch.arange(kv_length, device=device)
    if batch_size is None:
        batch_size = 1
    batch_arange = torch.arange(batch_size, device=device)
    head_arange = torch.arange(1, device=device)
    with TransformGetItemToIndex():
        mask = _vmap_for_bhqkv(mask_function)(batch_arange, head_arange, q_range, kv_arange)

    return mask

def flex_attention_mask(
    q_length: int,
    kv_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    mask_function: Callable = None,
    device: Optional[torch.device] = "cuda",
    **kwargs,
) -> BlockMask:
    """
    Create a BlockMask for flex attention given a token-level mask function.
    """
    kv_length = kv_length if kv_length is not None else q_length

    block_mask = create_block_mask(
        mask_mod=mask_function,
        B=batch_size,
        H=None,
        Q_LEN=q_length,
        KV_LEN=kv_length,
        device=device,
        _compile=True,
        BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    )

    return block_mask

def create_sliding_chunked_mask(
    length: int,
    attn_impl: str = "sdpa",
    chunk_size: int = None,
    sliding_window: int = 1,
    device: Optional[torch.device] = "cuda",
    **kwargs,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    """
    Create a sliding chunked mask based on the attention implementation used (sdpa | flex attention | sparge_attention).

    Args:
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        chunk_size (`int`):
            The size of the chunks to use for the causal mask.
        sliding_window (`int`):
            The size of the sliding window to use for the causal mask.
        attn_impl (`str`, *optional*, defaults to `"sdpa"`):
            The attention implementation to use. One of `"sdpa"`, `"flex_attention"` or `"sparge_attention"`.
    """
    mask_factory_function = sliding_chunked_mask_function(chunk_size, sliding_window)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[attn_impl]

    # We now create the mask
    mask = mask_interface(
        q_length=length,
        mask_function=mask_factory_function,
        device=device,
        )
    
    return mask

ALL_MASK_ATTENTION_FUNCTIONS = {
    "sdpa": sdpa_mask,
    "flex_attention": flex_attention_mask,
}
