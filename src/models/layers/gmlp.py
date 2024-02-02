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
    ):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(num_kernels, kernel_size))
        self.bias = nn.Parameter(torch.empty(num_kernels))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.kernel, std=1e-3)
        nn.init.constant_(self.bias, 1.0)

    def forward(self, x):
        # x: [B, D, L]
        B, D, L = x.shape
        C, K = self.kernel.shape
        assert D % C == 0, "num_channels must be divisible by num_kernels"

        kernel = self.kernel.repeat(D // C, 1)
        bias = self.bias.repeat(D // C)
        
        start = K - 1
        if K > self.__FFT_THRESHOLD:
            convolved = fft_conv1d(x, kernel)
        else:
            convolved = F.conv1d(x, kernel.flip(-1).view(D, 1, K), padding=K - 1, groups=D,)
        return convolved[:, :, start:(start + L)] + bias.view(-1, 1)

class GMLP(Layer):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        seq_len: int,
        ff_dim: int = -1,
        dropout=0.1,
        activation: ActivationFn = "relu",
        selection: ActivationFn = "softmax",
        layer_norm_eps=1e-05,
        mode: Literal["conv", "linear"] = "linear",
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
        if kernel_size == -1: kernel_size = 2 * seq_len + 1
        if ff_dim == -1: ff_dim = 4 * d_model

        # expansion
        self.layer_norm_1 = nn.LayerNorm(d_model, layer_norm_eps)
        self.ff_pointwise = nn.Sequential(
            nn.Linear(d_model, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim, layer_norm_eps, bias=False),
            nn.Dropout(dropout),
        )
        self.selection = build_activation(selection)

        # spatial mixin
        if mode == "conv":
            self.ff_spatial = nn.Sequential(
                SpatialConv(num_kernels, kernel_size),
                nn.LayerNorm(seq_len, layer_norm_eps, bias=False),
                nn.Dropout(dropout),
            )
        elif mode == "linear":
            self.ff_spatial = nn.Sequential(
                nn.Linear(seq_len, bottleneck_dim, bias=False),
                nn.Linear(bottleneck_dim, seq_len),
                nn.LayerNorm(seq_len, layer_norm_eps, bias=False),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        self.proj_out = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm_2 = nn.LayerNorm(d_model, layer_norm_eps)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            build_activation(activation),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model, layer_norm_eps, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        embeddings,  # [B, L, D]
        attention_mask=None,  # [B, L]
        token_type_ids=None,  # [B, L]
    ):
        embeddings = self.layer_norm_1(embeddings)

        attn_mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        embeddings = embeddings * attn_mask

        z = self.ff_pointwise(embeddings)  # [B, L, H]
        z = z * attn_mask
        z1, z2 = z.chunk(2, dim=-1)  # [B, L, H/2], [B, L, H/2]

        fwb_z2 = self.ff_spatial(z2.transpose(1, 2)).transpose(1, 2)

        s_z = self.selection(z1) * fwb_z2

        proj = embeddings + self.proj_out(s_z)

        return self.ff(self.layer_norm_2(proj)) + proj
