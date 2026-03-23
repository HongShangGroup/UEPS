import torch
import torch.nn as nn
Tensor = torch.Tensor


class PosEmbed(nn.Module):
    """
    Multiple strategies for additive position embedding with fixed sequence length.

    Shape:
        Output: [1, Ntok, width]
    """

    posemb_type_supported = ["learn", "sincos2d", "sincos1d", "absposri", "none"]

    def __init__(self, posemb_type: str, Ntok: int, width: int):
        super().__init__()

        assert posemb_type in self.posemb_type_supported
        self.posemb_type = posemb_type

        if posemb_type == "none":
            # No position embedding - register a zero buffer
            self.register_buffer('pos_embed', torch.zeros(1, Ntok, width), persistent=False)

        elif posemb_type == "learn":
            s = width ** -0.5
            self.pos_embed = nn.Parameter(s * torch.randn(1, Ntok, width))

        elif posemb_type == "sincos2d":
            self.pos_embed = nn.Parameter(torch.zeros(1, Ntok, width), requires_grad=False)
            self.pos_embed.data.copy_(
                torch.from_numpy(get_2d_sincos_pos_embed(width, int(Ntok ** 0.5))
                                 ).float().unsqueeze(0)
            )

        elif posemb_type == "sincos1d":
            self.pos_embed = nn.Parameter(torch.zeros(1, Ntok, width), requires_grad=False)
            self.pos_embed.data.copy_(get_1d_sincos_pos_embed(Ntok, width))

        elif posemb_type == "absposri":
            # assume tokens as [r0, r1, r2, ... i0, i1, i2, ...]
            # rt represent token from real(signal_t)
            # it represent token from imag(signal_t)
            assert (Ntok % 2) == 0
            N = Ntok // 2
            self.po_table = nn.Embedding(N, width)
            self.ri_table = nn.Embedding(2, width)
            po_idx = torch.cat((torch.arange(N), torch.arange(N)), dim=0).unsqueeze(0)
            ri_idx = torch.cat((torch.zeros(N, dtype=torch.int),
                                torch.ones(N, dtype=torch.int),
                                ), dim=0).unsqueeze(0)
            self.register_buffer('po_idx', po_idx, persistent=False)
            self.register_buffer('ri_idx', ri_idx, persistent=False)

    def forward(self):
        if self.posemb_type == "absposri":
            po_emb = self.po_table(self.po_idx)
            ri_emb = self.ri_table(self.ri_idx)
            x = po_emb + ri_emb
        else:
            x = self.pos_embed
        return x


import math
import numpy as np

# 1D sine-cosine position embedding as in
# Vaswani A, et al. Attention is all you need. NeurIPS 2017.

def get_1d_sincos_pos_embed(Ntok, width):
    angle_rads = np.arange(Ntok)[:, np.newaxis] / np.power(10000,
        (2 * (np.arange(width)[np.newaxis, :] // 2)) / np.float32(width))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pe = torch.from_numpy(angle_rads).float().unsqueeze(0)
    return pe

def get_1d_sincos_pos_embed_exp(Ntok, width):
    pe = torch.zeros(Ntok, width)
    position = torch.arange(Ntok, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe

# 2D sine-cosine position embedding from MAE
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

