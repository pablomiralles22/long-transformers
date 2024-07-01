import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.attention_head_handler import AttentionHeadHandler

class AttentionReducer(nn.Module):
    """
    Given input x: [B, L, D], reduce the sequence length by applying attention
    from a parameter query vector q: [D].
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout_p: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Parameter(torch.empty(d_model))

        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout_p = dropout_p

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.q, std=1e-3)

    def forward(self, x, attention_mask=None):
        # x: [B, L, D]
        # attention_mask: [B, L]
        B, L, D = x.shape

        Q = self.q.view(1, 1, D).repeat(B, 1, 1)  # [B, 1, D]
        K = self.W_K(x)
        V = self.W_V(x)
        
        Q = AttentionHeadHandler.separate_heads(Q, self.num_heads)  # [B, H, 1, D/H]
        K = AttentionHeadHandler.separate_heads(K, self.num_heads)  # [B, H, L, D/H]
        V = AttentionHeadHandler.separate_heads(V, self.num_heads)  # [B, H, L, D/H]

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, L]

        heads = F.scaled_dot_product_attention(Q, K, V, attention_mask, dropout_p=self.dropout_p)  # [B, H, 1, D/H]

        reduced = AttentionHeadHandler.join_heads(heads)  # [B, 1, D]

        return reduced.view(B, D)