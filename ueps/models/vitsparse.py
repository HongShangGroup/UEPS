import warnings

import torch
import torch.nn as nn
from typing import Optional
from functools import partial
from timm.models.vision_transformer import PatchEmbed
Tensor = torch.Tensor

from .embedding import PosEmbed
from .rope import VisionRotaryEmbeddingFast
from .common import RMSNorm, MLP, GatedMLP, UpSampLinear
from .attention_sparse import SUPPORTED_ATTENTION_IMPLS, ATTENTION_IMPLS
from .attention_mask import create_sliding_chunked_mask


class SparseAttention(nn.Module):
    """
    Attention module supporting various attention implementations.
    supported attention implementations:
    - sdpa: PyTorch's native scaled dot-product attention (SDPA)
    - flex_attention: Flex Attention
    - math: a simple PyTorch implementation of attention using matrix multiplications (not optimized, for testing only)
    supported local attention:
    - sliding_chunked_attention: chunked attention with sliding window (1D)
    """
    def __init__(
        self,
        dim: int,
        num_attention_heads: int = 8,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        q_bias: bool = True,
        k_bias: bool = True,
        v_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        use_rmsnorm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_type: str = "full_attention",
        attn_impl: str = "sdpa",
        sliding_window: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert dim % num_attention_heads == 0, "dim should be divisible by num_heads"

        assert attn_type in ("full_attention", "sliding_chunked_attention")
        self.attn_type = attn_type
        self.is_sliding_chunked = attn_type == "sliding_chunked_attention"

        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        assert (
            self.num_attention_heads % self.num_key_value_heads == 0
        ), "num_attention_heads must be divisible by num_key_value_heads"
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        if attn_type == "full_attention":
            self.attn_impl = attn_impl.lower() if attn_impl.lower() == "math" else "sdpa"
        else:
            self.attn_impl = attn_impl.lower()
        if self.attn_impl not in SUPPORTED_ATTENTION_IMPLS:
            warnings.warn(f"Unsupported attention implementation {attn_impl}, use sdpa instead")
            self.attn_impl = "sdpa"

        self.q_proj = nn.Linear(dim, self.num_attention_heads * self.head_dim, bias=q_bias)
        self.k_proj = nn.Linear(dim, self.num_key_value_heads * self.head_dim, bias=k_bias)
        self.v_proj = nn.Linear(dim, self.num_key_value_heads * self.head_dim, bias=v_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, dim, bias=proj_bias)

        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sliding_window = sliding_window if self.is_sliding_chunked else None
        self.chunk_size = chunk_size if self.is_sliding_chunked else None
        self.is_causal = False

    def forward(
        self,
        x: torch.Tensor,
        rope,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, patches, _  = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch_size, patches, -1, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, patches, -1, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, patches, -1, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        dropout_p = self.attn_drop.p if self.training else 0.0
        attn_output = ATTENTION_IMPLS[self.attn_impl](
            q,
            k,
            v,
            attention_mask=attention_mask,
            dropout=dropout_p if self.training else 0.0,
            sliding_window=self.sliding_window,
            chunk_size=self.chunk_size,
            module=self,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, patches, -1)
        attn_output = self.o_proj(attn_output)
        attn_output = self.proj_drop(attn_output)
        return attn_output


class ViTSparseBlock(nn.Module):
    """ViTe Block
    Adapted from torch.nn.transformer.TransformerEncoderLayer with changes below:
        no attention mask
        no dropout
        SwiGLU
        qknorm
        RMSNorm
    """
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 intermediate_size: int,
                 q_bias: bool = True,
                 k_bias: bool = True,
                 v_bias: bool = True,
                 proj_bias: bool = True,
                 eps: float = 1e-5,
                 use_swiglu: bool = False,
                 use_rmsnorm: bool = False,
                 use_qknorm: bool = False,
                 attn_type: str = "full_attention",
                 attn_impl: str = "sdpa",
                 sliding_window: Optional[int] = None,
                 chunk_size: Optional[int] = None,
                 ):
        super().__init__()

        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(d_model, eps=eps)
            self.norm2 = nn.LayerNorm(d_model, eps=eps)
        else:
            self.norm1 = RMSNorm(d_model, eps=eps)
            self.norm2 = RMSNorm(d_model, eps=eps)

        # Initialize attention layer
        self.attn = SparseAttention(
            d_model,
            nhead,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias,
            proj_bias=proj_bias,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            attn_type=attn_type,
            attn_impl=attn_impl,
            sliding_window=sliding_window,
            chunk_size=chunk_size,
        )
    
        # Initialize MLP layer
        intermediate_size = intermediate_size if intermediate_size is not None else 4 * d_model
        if use_swiglu:
            self.mlp = GatedMLP(d_model, intermediate_size)
        else:
            act_layer = lambda: nn.GELU()
            self.mlp = MLP(
                in_features=d_model,
                hidden_features=intermediate_size,
                act_layer=act_layer,
            )

    def forward(self, x: Tensor, rope=None, attention_mask=None) -> Tensor:
        residual = x
        x = self.norm1(x)
        x = self.attn(x, rope=rope, attention_mask=attention_mask)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class ViTSparse(nn.Module):
    """
    A ViTe based model supporting sparse attention.

    Shape:
        Input: [N, C, H, H]
        Output: [N, C, H, H]
    """

    config_dict = {
        "tiny":   {"embed_dim": 128,  "layers": 6,  "heads": 4},
        "small":  {"embed_dim": 384,  "layers": 12, "heads": 6},
        "base":   {"embed_dim": 512,  "layers": 18, "heads": 8},
        "large":  {"embed_dim": 768,  "layers": 24, "heads": 12},
        "xlarge": {"embed_dim": 1024, "layers": 24, "heads": 16},
    }

    def __init__(self,
                 H: int = 128,
                 Cin: int = 1,
                 Cout: int = 1,
                 patch_size: int = 8,
                 posemb_type: str = "sincos2d",
                 attention_config: str = "base",
                 width: int = None,
                 heads: int = None,
                 layers: int = None,
                 intermediate_size: int = None,
                 q_bias: bool = True,
                 k_bias: bool = True,
                 v_bias: bool = True,
                 proj_bias: bool = True,
                 eps: float = 1e-5,
                 use_qknorm: bool = False,
                 use_swiglu: bool = False,
                 use_rope: bool = False,
                 use_rmsnorm: bool = False,
                 attn_impl: str = "sdpa",
                 attn_types: list = None,
                 sliding_window: Optional[int] = None,
                 chunk_size: Optional[int] = None,
                 ):
        super().__init__()

        assert attention_config in list(self.config_dict.keys())
        if width is None:
            width = self.config_dict[attention_config]["embed_dim"]
        if heads is None:
            heads = self.config_dict[attention_config]["heads"]
        if layers is None:
            layers = self.config_dict[attention_config]["layers"]
        self.width = width
        self.heads = heads
        self.layers = layers

        if attn_types is None:
            attn_types = ["full_attention"] * layers
        else:
            assert len(attn_types) == layers, "attn_types length must match number of layers"

        if use_rmsnorm:
            norm_layer = partial(RMSNorm, eps=eps)
        else:
            norm_layer = partial(nn.LayerNorm, eps=eps, elementwise_affine=False)
        self.patchemb = PatchEmbed(H, patch_size, Cin, width, bias=True,
                                   norm_layer=norm_layer)

        if use_rope:
            half_head_dim = width // heads // 2
            hw_seq_len = H // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            if posemb_type == "none":
                raise Warning('posemb should not be none when not using rope')
            self.feat_rope = None

        L = self.patchemb.num_patches
        self.pos_embed = PosEmbed(posemb_type, L, width)

        # 2D grid dimensions derived from image size and patch size
        grid_h = H // patch_size
        grid_w = H // patch_size
        
        self.attn_impl = attn_impl
        self.is_sliding_chunked = "sliding_chunked_attention" in attn_types if attn_types is not None else False

        # Create the masks
        mask_kwargs = {
                "length": L,
                "attn_impl": self.attn_impl,
                "sliding_window": sliding_window,
                "chunk_size": chunk_size,
            }
        self.mask_mapping = {
            "full_attention": None,
            "sliding_chunked_attention": create_sliding_chunked_mask(**mask_kwargs) if self.is_sliding_chunked else None,
        }

        self.use_swiglu = use_swiglu
        self.blocks = nn.ModuleList([
            ViTSparseBlock(
                 width,
                 heads,
                 intermediate_size,
                 q_bias=q_bias,
                 k_bias=k_bias,
                 v_bias=v_bias,
                 proj_bias=proj_bias,
                 eps=eps,
                 use_swiglu=use_swiglu,
                 use_rmsnorm=use_rmsnorm,
                 use_qknorm=use_qknorm,
                 attn_type=attn_types[layer_idx] if attn_types is not None else "full_attention",
                 attn_impl=attn_impl,
                 sliding_window=sliding_window,
                 chunk_size=chunk_size,
                 ) for layer_idx in range(layers)
        ])

        self.final_layers = UpSampLinear(width, L, patch_size, Cout)

        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch embed
        w = self.patchemb.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patchemb.proj.bias, 0)

        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.q_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.k_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.v_proj.weight, std=attn_std)
            nn.init.normal_(block.mlp.w12.weight, std=fc_std)
            nn.init.normal_(block.mlp.w3.weight, std=proj_std)

        # Zero-out transformer layers
        for block in self.blocks:
            nn.init.constant_(block.attn.o_proj.weight, 0)
            nn.init.constant_(block.attn.o_proj.bias, 0)
            nn.init.constant_(block.mlp.w3.weight, 0)
            nn.init.constant_(block.mlp.w3.bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layers.linear.weight, 0)
        nn.init.constant_(self.final_layers.linear.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patchemb(x) + self.pos_embed()
        
        for block in self.blocks:
            x = block(x, self.feat_rope, attention_mask=self.mask_mapping[block.attn.attn_type])

        x = self.final_layers(x)
        return x
