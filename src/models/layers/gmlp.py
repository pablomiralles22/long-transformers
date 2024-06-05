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
        num_channels: int,
        kernel_size: int,
        direction: Literal["forward", "backward", "bidirectional"] = "bidirectional",
    ):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(num_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(num_channels))
        self.direction = direction
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.kernel, std=1e-5)
        nn.init.constant_(self.bias, 1.0)

    def forward(self, x):
        # x: [B, D, L]
        B, D, L = x.shape
        C, K = self.kernel.shape
        assert D % C == 0, "num_channels must be divisible by num_channels"

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

class SpatialLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        num_hidden_layers: int = 0,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, bottleneck_dim)
        self.linear_2 = nn.Linear(bottleneck_dim, input_dim)

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(bottleneck_dim, bottleneck_dim, bias=False),
                nn.LayerNorm(bottleneck_dim),
                build_activation("gelu"),
            )
            for _ in range(num_hidden_layers)
        ])
    
    def forward(self, x):
        # x: [B, D, L]
        y = F.gelu(self.linear_1(x))
        for layer in self.hidden_layers:
            y = layer(y) + y
        return self.linear_2(y)
    
class LinearBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout=0.1,
        activation: ActivationFn = "gelu",
        norm_eps=1e-05,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, norm_eps)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.ff(self.norm(x)) + x
        

class GMLP(Layer):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        seq_len: int,
        dropout=0.1,
        activation: ActivationFn = "gelu",
        selection: ActivationFn = "gelu",
        norm_eps=1e-05,
        mode: Literal["conv", "linear"] = "linear",
        direction: Literal["forward", "backward", "bidirectional"] = "bidirectional",
        residual: bool = True,
        # conv mode params
        kernel_size: int = -1,
        num_channels: int = -1,
        # linear mode params
        bottleneck_dim: int = -1,
        num_hidden_layers: int = 3,
        # ff block params
        ff_hidden_dim: int = None,
    ):
        super().__init__()

        # set default values
        if bottleneck_dim == -1: bottleneck_dim = seq_len // 2
        if num_channels == -1: num_channels = hidden_dim
        if kernel_size == -1:
            kernel_size = (2 * seq_len + 1) if direction == "bidirectional" else seq_len

        self.residual = residual
        self.hidden_dim = hidden_dim

        # expansion
        self.norm_1 = nn.LayerNorm(d_model, norm_eps)
        self.ff_z2 = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            build_activation(activation),
            # nn.LayerNorm(hidden_dim, norm_eps),
        )
        self.ff_z1 = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            build_activation(selection),
        )

        # spatial mixin
        if mode == "conv":
            self.spatial_layer = SpatialConv(num_channels, kernel_size, direction=direction)
        elif mode == "linear":
            self.spatial_layer = SpatialLinear(seq_len, bottleneck_dim, num_hidden_layers)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        self.norm_2 = nn.LayerNorm(hidden_dim, norm_eps)

        self.proj_out = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

        if ff_hidden_dim is not None:
            self.ff_block = LinearBlock(d_model, ff_hidden_dim, dropout, activation, norm_eps)

    def forward(
        self,
        embeddings,  # [B, L, D]
        attention_mask=None,  # [B, L]
        token_type_ids=None,  # [B, L]
    ):
        x = self._forward_gmlp_block(embeddings, attention_mask)
        x = self._forward_ff_block(x)
        return x
    
    def _forward_gmlp_block(
        self,
        embeddings,  # [B, L, D]
        attention_mask=None,  # [B, L]
    ):
        x = self.norm_1(embeddings)

        z1 = self.ff_z1(x)  # [B, L, H]
        z2 = self.ff_z2(x)  # [B, L, H]

        attn_mask = attention_mask.unsqueeze(-1).bool()  # [B, L, 1]
        z2 = z2.masked_fill(~attn_mask, 0.)

        fwb_z2 = self.spatial_layer(z2.transpose(1, 2)).transpose(1, 2)  # [B, L, H]
        s_z = z1 * self.norm_2(fwb_z2)

        if self.residual is True:
            return self.proj_out(s_z) + embeddings
        return self.proj_out(s_z)
    
    def _forward_ff_block(self, x):
        if self.ff_block is None:
            return x
        if self.residual is True:
            return self.ff_block(x) + x
        return self.ff_block(x)