#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from attention_module import pair, PreNorm, PostNorm, StandardAttention, LinearAttention, ReLUFeedForward, FeedForward


class TransformerCatNoCls(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 use_ln=False,
                 scale=16,     # can be list, or an int
                 dropout=0.,
                 relative_emb_dim=2,
                 min_freq=1/100,
                 attention_init='orthogonal',
                 init_gain=None,
                 use_relu=False,
                 cat_pos=False,
                 ):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth
        print(f"Check if len(scale) equals to depth")
        print(f"depth: {depth}")
        print(f"scale: {scale}, length of scale: {len(scale)}")
        assert len(scale) == depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.use_ln = use_ln

        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList([
                        PreNorm(dim, StandardAttention(dim, heads=heads,
                                dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim,  FeedForward(dim, mlp_dim, dropout=dropout)
                                if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout))]),
                )
        else:
            for d in range(depth):
                if scale[d] != -1 or cat_pos:
                    attn_module = LinearAttention(dim, attn_type,
                                                  heads=heads, dim_head=dim_head, dropout=dropout,
                                                  relative_emb=True, scale=scale[d],
                                                  relative_emb_dim=relative_emb_dim,
                                                  min_freq=min_freq,
                                                  init_method=attention_init,
                                                  init_gain=init_gain
                                                  )
                else:
                    attn_module = LinearAttention(dim, attn_type,
                                                  heads=heads, dim_head=dim_head, dropout=dropout,
                                                  cat_pos=True,
                                                  pos_dim=relative_emb_dim,
                                                  relative_emb=False,
                                                  init_method=attention_init,
                                                  init_gain=init_gain
                                                  )
                if not use_ln:  # If depth is 4: there are 4 attention modules and 4 feedforward modules in total, making up 4 transformer layers.
                    # Each transformer layer consists of one attention module and one feedforward module
                    self.layers.append(
                        nn.ModuleList([
                            attn_module,
                            FeedForward(dim, mlp_dim, dropout=dropout)
                            if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout)
                        ]),
                    )
                else:
                    self.layers.append(
                        nn.ModuleList([
                            nn.LayerNorm(dim),
                            attn_module,
                            nn.LayerNorm(dim),
                            FeedForward(dim, mlp_dim, dropout=dropout)
                            if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout),
                        ]),
                    )

    def forward(self, x, pos_embedding):
        # x in [b n c], pos_embedding in [b n 2]
        b, n, c = x.shape

        for layer_no, attn_layer in enumerate(self.layers):
            if not self.use_ln:
                [attn, ffn] = attn_layer

                x = attn(x, pos_embedding) + x
                x = ffn(x) + x
            else:
                [ln1, attn, ln2, ffn] = attn_layer
                x = ln1(x)
                x = attn(x, pos_embedding) + x
                x = ln2(x)
                x = ffn(x) + x
        return x


class SpatialTemporalEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 # embedding dim of token                 (how about 512)
                 in_emb_dim,
                 # embedding dim of encoded sequence      (how about 256)
                 out_seq_emb_dim,
                 heads,
                 # depth of transformer / how many layers of attention
                 depth,
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        if depth > 4:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=[32, 16, 8, 8] +
                                                                             [1] *
                                                     (depth - 4),
                                                     attention_init='orthogonal')
        else:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=[32] + [16]*(depth-2) + [1],
                                                     attention_init='orthogonal')

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, t(*c)+2, n]
                input_pos,  # [b, n, 2]
                ):

        x = self.to_embedding(x)
        x = self.s_transformer.forward(x, input_pos)
        x = self.project_to_latent(x)

        return x


class SpatialEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 # embedding dim of token                 (how about 512)
                 in_emb_dim,
                 # embedding dim of encoded sequence      (how about 256)
                 out_seq_emb_dim,
                 heads,
                 # depth of transformer / how many layers of attention    (4)
                 depth,
                 res,
                 use_ln=True,
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                 'galerkin',
                                                 use_ln=use_ln,
                                                 scale=[res, res//4] +
                                                 [1]*(depth-2),
                                                 relative_emb_dim=2,
                                                 min_freq=1 / res,
                                                 attention_init='orthogonal')

        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, n, c]
                input_pos,  # [b, n, 2]
                ):

        x = self.to_embedding(x)
        x = self.s_transformer.forward(x, input_pos)
        x = self.to_out(x)

        return x
