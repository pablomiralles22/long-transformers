import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal
from src.models.layers.activations import ActivationFn, build_activation
from src.models.layers.layer import Layer
from src.models.functional.fft_conv1d import fft_conv1d

class SpatialConv(nn.Module):
    __FFT_THRESHOLD: int = 30

    def __init__(
        self,
        num_kernels: int,
        kernel_size: int,
        direction: Literal["forward", "backward", "bidirectional"] = "bidirectional",
    ):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(num_kernels, kernel_size))
        self.bias = nn.Parameter(torch.empty(num_kernels))
        self.direction = direction
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.kernel, std=1e-4)
        nn.init.constant_(self.bias, 1.0)

    def forward(self, x):
        # x: [B, D, L]
        B, D, L = x.shape
        C, K = self.kernel.shape
        assert D % C == 0, "num_channels must be divisible by num_kernels"

        # kernel = self.kernel.repeat(D // C, 1)
        kernel = self.kernel
        bias = self.bias.repeat(D // C)
        
        if self.direction == "forward":
            start = K - 1
        elif self.direction == "backward":
            start = 0
        elif self.direction == "bidirectional":
            start = K // 2

        if K > self.__FFT_THRESHOLD:
            convolved = fft_conv1d(x, kernel)
        else:
            kernel = kernel.flip(-1).repeat(D // C, 1).view(D, 1, K)
            convolved = F.conv1d(x, kernel, padding=K-1, groups=D,)

        return convolved[:, :, start:(start + L)] + bias.view(-1, 1)

class GMLP(Layer):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        seq_len: int,
        dropout=0.1,
        activation: ActivationFn = "silu",
        selection: ActivationFn = "sigmoid",
        norm_eps=1e-05,
        mode: Literal["conv", "linear"] = "linear",
        direction: Literal["forward", "backward", "bidirectional"] = "bidirectional",
        residual: bool = True,
        # conv mode params
        num_kernels: int = -1,
        kernel_size: int = -1,
        # linear mode params
        bottleneck_dim: int = -1,
    ):
        super().__init__()

        # set default values
        if bottleneck_dim == -1: bottleneck_dim = seq_len // 2
        if num_kernels == -1: num_kernels = hidden_dim
        if kernel_size == -1:
            kernel_size = (2 * seq_len + 1) if direction == "bidirectional" else seq_len

        self.residual = residual
        self.hidden_dim = hidden_dim

        # if self.residual is True:
        #     self.residual_proj = nn.Sequential(
        #         nn.Linear(d_model, d_model, bias=False),
        #         nn.LayerNorm(d_model, norm_eps),
        #     )

        # expansion
        self.norm_1 = nn.LayerNorm(d_model, norm_eps)
        self.ff_pointwise = nn.Sequential(
            nn.Linear(d_model, 2 * hidden_dim, bias=False),
            nn.LayerNorm(2 * hidden_dim, norm_eps),
            build_activation(activation),
        )

        # spatial mixin
        spatial_dim = seq_len
        if mode == "conv":
            spatial_layers = [ SpatialConv(num_kernels, kernel_size, direction=direction) ]
        elif mode == "linear":
            spatial_layers = [ nn.Linear(spatial_dim, bottleneck_dim), nn.Linear(bottleneck_dim, spatial_dim) ]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        spatial_layers += [ nn.BatchNorm1d(hidden_dim, norm_eps, affine=False), ]
        self.ff_spatial = nn.Sequential(*spatial_layers)
        self.norm_2 = nn.LayerNorm(hidden_dim, norm_eps)

        self.proj_out = nn.Sequential(
            nn.Linear(hidden_dim, d_model, bias=False),
            nn.LayerNorm(d_model, norm_eps),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        embeddings,  # [B, L, D]
        attention_mask=None,  # [B, L]
        token_type_ids=None,  # [B, L]
    ):
        x = self.norm_1(embeddings)

        attn_mask = attention_mask.unsqueeze(-1).bool()  # [B, L, 1]

        z = self.ff_pointwise(x)  # [B, L, 2H]
        z.masked_fill_(~attn_mask, 0.)

        z1, z2 = z.chunk(2, dim=-1)  # [B, L, H], [B, L, H]

        fwb_z2 = self.ff_spatial(z2.transpose(1, 2)).transpose(1, 2)  # [B, L, H]
        s_z = z1 * fwb_z2

        if self.residual is True:
            return self.proj_out(s_z) + embeddings
            # return self.proj_out(s_z) + embeddings + self.residual_proj(embeddings)
            # return self.proj_out(s_z) + self.residual_proj(embeddings)

        return self.proj_out(s_z)

    # def forward(
    #     self,
    #     embeddings,  # [B, L, D]
    #     attention_mask=None,  # [B, L]
    #     token_type_ids=None,  # [B, L]
    # ):
    #     embeddings = self.norm_1(embeddings)

    #     attn_mask = attention_mask.unsqueeze(-1).bool()  # [B, L, 1]

    #     y = self.ff_pointwise(embeddings)  # [B, L, 2H]
    #     y.masked_fill_(~attn_mask, 0.)

    #     # z1, z2 = z.chunk(2, dim=-1)  # [B, L, H], [B, L, H]
    #     y1, y2 = y.split([2 * self.hidden_dim, self.hidden_dim], dim=-1)  # [B, L, 2H], [B, L, H]

    #     fwb_y2 = self.ff_spatial(y2.transpose(1, 2)).transpose(1, 2)  # [B, L, H]
    #     # s_z = self.selection(z1) * fwb_z2
    #     # s_z = self.selection(fwb_z2) * z1
    #     z = y1 + self.spatial_expand(fwb_y2)  # [B, L, 2H]
    #     z1, z2 = z.chunk(2, dim=-1)  # [B, L, H]
    #     s_z = self.selection(z1) * z2

    #     if self.residual is True:
    #          return self.proj_out(s_z) + embeddings
    #     return self.proj_out(s_z)