import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal
from src.models.layers.activations import ActivationFn, build_activation
from src.models.layers.layer import Layer
from src.models.functional.fft_conv1d import fft_conv1d

class BatchNorm1dTranspose(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.bn(x)
        return x.transpose(-1, -2)

class SpatialConv(nn.Module):
    __FFT_THRESHOLD: int = 30

    def __init__(
        self,
        num_kernels: int,
        kernel_size: int,
        causal: bool = False,
    ):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(num_kernels, kernel_size))
        self.bias = nn.Parameter(torch.empty(num_kernels))
        self.causal = causal
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.kernel, std=1e-3)
        nn.init.constant_(self.bias, 1.0)

    def forward(self, x):
        # x: [B, D, L]
        B, D, L = x.shape
        C, K = self.kernel.shape
        assert D % C == 0, "num_channels must be divisible by num_kernels"

        # kernel = self.kernel.repeat(D // C, 1)
        kernel = self.kernel
        bias = self.bias.repeat(D // C)
        
        start = K - 1
        if K > self.__FFT_THRESHOLD:
            convolved = fft_conv1d(x, kernel)
        else:
            kernel = kernel.flip(-1).repeat(D // C, 1).view(D, 1, K)
            convolved = F.conv1d(x, kernel, padding=K - 1, groups=D,)
        return convolved[:, :, start:(start + L)] + bias.view(-1, 1)

class GMLP(Layer):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        seq_len: int,
        dropout=0.1,
        activation: ActivationFn = "relu",
        selection: ActivationFn = "softmax",
        norm_eps=1e-05,
        mode: Literal["conv", "linear"] = "linear",
        causal: bool = False,
        residual: bool = True,
        # conv mode params
        num_kernels: int = -1,
        kernel_size: int = -1,
        # linear mode params
        bottleneck_dim: int = -1,
    ):
        super().__init__()

        assert not (causal and mode == "linear"), "Causal mode is only supported for conv mode"

        # set default values
        if bottleneck_dim == -1: bottleneck_dim = seq_len // 2
        if num_kernels == -1: num_kernels = hidden_dim
        if kernel_size == -1: kernel_size = (2 * seq_len + 1)

        self.residual = residual

        # expansion
        self.batch_norm_1 = BatchNorm1dTranspose(d_model, norm_eps)
        self.ff_pointwise = nn.Sequential(
            nn.Linear(d_model, 2 * hidden_dim, bias=False),
            BatchNorm1dTranspose(2 * hidden_dim, norm_eps),
            nn.Dropout(dropout),
        )
        self.selection = build_activation(selection)

        # spatial mixin
        spatial_dim = seq_len
        if mode == "conv":
            spatial_layers = [ SpatialConv(num_kernels, kernel_size, causal=causal) ]
        elif mode == "linear":
            spatial_layers = [ nn.Linear(spatial_dim, bottleneck_dim), nn.Linear(bottleneck_dim, spatial_dim) ]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        spatial_layers += [
            nn.BatchNorm1d(hidden_dim, norm_eps, affine=False),
            nn.Dropout(dropout),
        ]
        self.ff_spatial = nn.Sequential(*spatial_layers)

        self.proj_out = nn.Sequential(
            nn.Linear(hidden_dim, d_model, bias=False),
            BatchNorm1dTranspose(d_model, norm_eps),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        embeddings,  # [B, L, D]
        attention_mask=None,  # [B, L]
        token_type_ids=None,  # [B, L]
    ):
        embeddings = self.batch_norm_1(embeddings)

        attn_mask = attention_mask.unsqueeze(-1).bool()  # [B, L, 1]

        z = self.ff_pointwise(embeddings)  # [B, L, 2H]
        z.masked_fill_(~attn_mask, 0.)

        z1, z2 = z.chunk(2, dim=-1)  # [B, L, H], [B, L, H]

        fwb_z2 = self.ff_spatial(z2.transpose(1, 2)).transpose(1, 2)  # [B, L, H]
        s_z = self.selection(z1) * fwb_z2
        # s_z = self.selection(fwb_z2) * z1

        if self.residual is True:
             return self.proj_out(s_z) + embeddings
        return self.proj_out(s_z)