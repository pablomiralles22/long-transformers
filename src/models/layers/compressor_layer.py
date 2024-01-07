import torch

from torch import nn
from torch.nn import functional as F
from typing import Literal
from src.models.modules.ema import EMA


class CompressorStep(nn.Module):
    def __init__(
        self,
        d_model,  # D1
        vdim=None,  # D2
        nhead=4,
        dropout=0.1,
        dim_feedforward=2048,
        activation_fn_cls=nn.ReLU,
        layer_norm_eps=1e-05,
        norm_first=True,
    ):
        super().__init__()

        self.norm_first = norm_first

        # multihead attention
        self.mh_attention = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=True,
            dropout=dropout,
            vdim=vdim,
            kdim=vdim,
        )

        # dropout after attention
        self.dropout = nn.Dropout(dropout)

        # layer norms
        if self.norm_first is True:
            self.layer_norm_1_1 = nn.LayerNorm(d_model, layer_norm_eps)
            self.layer_norm_1_2 = nn.LayerNorm(vdim, layer_norm_eps)
        else:
            self.layer_norm_1 = nn.LayerNorm(d_model, layer_norm_eps)

        self.layer_norm_2 = nn.LayerNorm(d_model, layer_norm_eps)

        # feed forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            activation_fn_cls(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        embeddings_1,  # [B, L1, D1]; query
        embeddings_2,  # [B, L2, D2]; key, value
        attention_mask=None,  # [B, L2]; key
    ):
        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()

        if self.norm_first is True:
            embeddings_1 = self.layer_norm_1_1(embeddings_1)  # [B, L1, D1]
            embeddings_2 = self.layer_norm_1_2(embeddings_2)  # [B, L2, D2]

            x, _ = self.mh_attention(
                embeddings_1,
                embeddings_2,
                embeddings_2,
                key_padding_mask=attention_mask,
                need_weights=False,
            )  # [B, L1, D1]
            x = embeddings_1 + self.dropout(x)  # [B, L1, D1]
            x = x + self.ff(self.layer_norm_2(x))  # [B, L1, D1]
        else:
            x, _ = self.mh_attention(
                embeddings_1,
                embeddings_2,
                embeddings_2,
                key_padding_mask=attention_mask,
                need_weights=False,
            )  # [B, L1, D1]
            x = self.dropout(x)
            x = self.layer_norm_1(embeddings_1 + x)  # [B, L1, D1]
            y = self.ff(x)  # [B, L1, D1]
            x = self.layer_norm_2(x + y)  # [B, L1, D1]

        return x


class CompressorLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=4,
        mem_dim=None,
        mem_nhead=None,
        dropout=0.1,
        dim_feedforward=2048,
        activation_fn_cls=nn.ReLU,
        layer_norm_eps=1e-05,
        use_ema: bool = False,
        ema_dim: int = 2,
        ema_direction: Literal["forward", "backward", "bidirectional"] = "forward",
        conv_size: int = 100,
    ):
        super().__init__()
        mem_dim = mem_dim if mem_dim is not None else d_model
        mem_nhead = mem_nhead if mem_nhead is not None else nhead

        self.compress_step = CompressorStep(
            mem_dim,
            vdim=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            activation_fn_cls=activation_fn_cls,
            layer_norm_eps=layer_norm_eps,
        )
        self.update_step = CompressorStep(
            d_model,
            vdim=mem_dim,
            nhead=mem_nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            activation_fn_cls=activation_fn_cls,
            layer_norm_eps=layer_norm_eps,
        )

        if use_ema is True:
            self.ema = EMA(d_model, ema_dim, conv_size, d_model, direction=ema_direction)
            # self.ema_ff = nn.Sequential(
            #     activation_fn_cls(),
            #     nn.Linear(d_model, d_model),
            #     nn.LayerNorm(d_model, layer_norm_eps),
            #     nn.Dropout(dropout),
            # )

    def forward(
        self,
        embeddings,  # [B, SL, D]
        memory,  # [B, ML, D]
        attention_mask=None,  # [B, SL]
    ):
        if hasattr(self, "ema"):
            embeddings = embeddings + F.silu(self.ema(embeddings))
        
        updated_mem = self.compress_step(
            memory, embeddings, attention_mask=attention_mask
        )
        updated_embeddings = self.update_step(embeddings, updated_mem)
        return updated_embeddings, updated_mem


class Compressor(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=4,
        mem_length=128,
        mem_dim=None,
        mem_nhead=None,
        num_layers=1,
        dropout=0.1,
        dim_feedforward=2048,
        activation_fn_cls=nn.ReLU,
        layer_norm_eps=1e-05,
        use_ema: bool = False,
        ema_dim: int = 2,
        ema_direction: Literal["forward", "backward", "bidirectional"] = "forward",
        conv_size: int = 100,
        shared_layers=False,
    ):
        super().__init__()

        if shared_layers is True:
            self.compressor_layers = nn.ModuleList(
                [
                    CompressorLayer(
                        d_model, nhead=nhead,
                        mem_dim=mem_dim, mem_nhead=mem_nhead,
                        dropout=dropout,
                        dim_feedforward=dim_feedforward,
                        activation_fn_cls=activation_fn_cls,
                        layer_norm_eps=layer_norm_eps,
                        use_ema=use_ema, ema_dim=ema_dim, ema_direction=ema_direction, conv_size=conv_size,
                    )
                ] * num_layers
            )
        else:
            self.compressor_layers = nn.ModuleList(
                [
                    CompressorLayer(
                        d_model, nhead=nhead,
                        mem_dim=mem_dim, mem_nhead=mem_nhead,
                        dropout=dropout,
                        dim_feedforward=dim_feedforward,
                        activation_fn_cls=activation_fn_cls,
                        layer_norm_eps=layer_norm_eps,
                        use_ema=use_ema, ema_dim=ema_dim, ema_direction=ema_direction, conv_size=conv_size,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.memory = nn.Parameter(torch.empty(mem_length, mem_dim or d_model))
        nn.init.xavier_uniform_(self.memory)

    def forward(
        self,
        embeddings,  # [B, L, D]
        attention_mask=None,  # [B, L]
        token_type_ids=None,  # [B, L]
    ):
        x = embeddings
        memory = self.memory.unsqueeze(0).repeat(
            embeddings.shape[0], 1, 1
        )
        for layer in self.compressor_layers:
            x, memory = layer(x, memory, attention_mask=attention_mask)
        return x
