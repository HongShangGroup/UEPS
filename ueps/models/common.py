import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional
Tensor = torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = in_features
        self.intermediate_size = hidden_features or in_features
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = act_layer() if act_layer else F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.act_fn = act_layer() if act_layer else F.silu
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias) 

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = self.act_fn(x1) * x2
        return self.w3(hidden)


class UpSampLinear(nn.Module):
    """
    Reshape output tokens to get 2D layout
        reshape all tokens to a h*h grid (L = h*h)
        reshape each token to a p*p grid, each with channel c
        final grid size H = h*p, each with channel c

    reshape code is adpated from DiT
    https://github.com/facebookresearch/DiT/blob/main/models.py

    Args:
        width: int, transformer hidden size
        L: int, number of output tokens
        p: int, unpatch size
        c: int, output channels

    Shape:
        Input: [Nbs, L, width]
        Output: [Nbs, c, H, H]
    """
    def __init__(self, width, L, p, c):
        super().__init__()
        self.norm_final = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(width, p * p * c, bias=True)
        h = w = int(L ** 0.5)
        self.h = h
        self.w = w
        self.p = p
        self.c = c

    def forward(self, x):
        x = self.linear(self.norm_final(x))
        x = x.reshape(shape=(x.shape[0], self.h, self.w, self.p, self.p, self.c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], self.c, self.h * self.p, self.h * self.p))
        return x
