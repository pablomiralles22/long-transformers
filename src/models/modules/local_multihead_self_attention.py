import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from src.utils.rotary_embedding import RotaryEmbedding
from src.utils.attention_head_handler import AttentionHeadHandler
from src.utils.attention_window_builder import AttentionWindowBuilder

class LocalMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        window_size: int,
        bias: bool = True,
        vdim: Optional[int] = None,
        qkdim: Optional[int] = None,
        freq: int = 10000,
    ):
        assert d_model % nhead == 0, "Error: the embedding dimension should be divisible by the number of heads"

        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        self.vdim = d_model if vdim is None else vdim
        self.qkdim = d_model if qkdim is None else qkdim

        dim_qkv = self.qkdim * 2 + self.vdim
        self.W_QKV = nn.Linear(d_model, dim_qkv, bias=bias)
        self.WO = nn.Linear(self.vdim, d_model, bias=bias)

        self.window_size = window_size
        self.freq = freq

    def forward(
        self,
        embeddings,  # (B, L, D)
        attention_mask=None,  # (B, L)
    ):  # -> (B, L, D)
        _, L, D = embeddings.size()
        device = embeddings.device

        qkv = self.W_QKV(embeddings)  # (B, L, 2 * QKDIM + VDIM)
        queries, keys, values = qkv.split(
            [self.qkdim, self.qkdim, self.vdim],
            dim=-1,
        )  # (B, L, QKDIM), (B, L, QKDIM), (B, L, VDIM)

        queries = AttentionHeadHandler.separate_heads(queries, self.nhead)  # (B, H, L, QKDIM)
        keys = AttentionHeadHandler.separate_heads(keys, self.nhead)  # (B, H, L, QKDIM)
        values = AttentionHeadHandler.separate_heads(values, self.nhead)  # (B, H, L, VDIM)

        queries = RotaryEmbedding.apply(queries, freq=self.freq)  # (B, H, L, QKDIM)
        keys = RotaryEmbedding.apply(keys, freq=self.freq)  # (B, H, L, QKDIM)

        # local attention
        idxs = AttentionWindowBuilder.build_idxs(self.window_size, L, device=device)  # [1, L, W]
        attention_scores = torch.einsum(
            'b h l w d , b h l d -> b h l w', 
            keys[..., idxs, :],
            queries,
        ) / (D ** 0.5)  # [B, H, L, W]

        if attention_mask is not None:
            windowed_attention_mask = (
                AttentionWindowBuilder
                .build_attention_mask(self.window_size, L, attention_mask.bool(), device=device)  # [B, L, W]
                .unsqueeze(1)  # [B, 1, L, W]
            )
            attention_scores.masked_fill_(~windowed_attention_mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)  # [B, H, L, W]
        outputs = torch.einsum('b h l w , b h l w d -> b h l d', attention_weights, values[..., idxs, :])  # [B, H, L, D]

        # global attention for token CLS (idx=0)
        cls_attention_scores = torch.einsum(
            'b h d , b h l d -> b h l', 
            queries[..., 0, :],  # [B, H, D]
            keys,  # [B, H, L, D]
        ) / (D ** 0.5)  # [B, H, L]
        cls_attention_scores = cls_attention_scores.masked_fill(~attention_mask.bool().unsqueeze(1), float('-inf'))  # [B, H, L]
        cls_attention_weights = F.softmax(cls_attention_scores, dim=-1)  # [B, H, L]
        cls_outputs = torch.einsum('b h l , b h l d -> b h d', cls_attention_weights, values)  # [B, H, D]

        outputs[..., 0, :] = cls_outputs
        
        return self.WO(AttentionHeadHandler.join_heads(outputs))
