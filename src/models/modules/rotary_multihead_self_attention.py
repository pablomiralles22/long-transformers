import torch.nn as nn
import torch.nn.functional as F

from src.utils.rotary_embedding import RotaryEmbedding
from src.utils.attention_head_handler import AttentionHeadHandler

class RotaryMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        bias=True,
        vdim=None,
        qkdim=None,
        freq=10000,
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

        self.freq = freq

    def forward(
        self,
        embeddings,  # (B, L, D)
        attention_mask=None,  # (B, L)
    ):  # -> (B, L, D)
        QKV = self.W_QKV(embeddings)  # (B, L, 2 * QKDIM + VDIM)
        Q, K, V = QKV.split(
            [self.qkdim, self.qkdim, self.vdim],
            dim=-1,
        )  # (B, L, QKDIM), (B, L, QKDIM), (B, L, VDIM)

        Q = AttentionHeadHandler.separate_heads(Q, self.nhead)  # (B, H, L, QKDIM)
        K = AttentionHeadHandler.separate_heads(K, self.nhead)  # (B, H, L, QKDIM)
        V = AttentionHeadHandler.separate_heads(V, self.nhead)  # (B, H, L, VDIM)

        Q = RotaryEmbedding.apply(Q, freq=self.freq)  # (B, H, L, QKDIM)
        K = RotaryEmbedding.apply(K, freq=self.freq)  # (B, H, L, QKDIM)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()  # (B, 1, 1, L)
        heads = F.scaled_dot_product_attention(Q, K, V, attn_mask=attention_mask)  # (...B, H, L, VDIM)
        return self.WO(AttentionHeadHandler.join_heads(heads))
