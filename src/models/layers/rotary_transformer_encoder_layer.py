import torch
import math

from torch import nn
from torch.nn import functional as F


class RotaryEmbedding:
    def __init__(self):
        self.cache = dict()

    def apply(self, X):
        # X: (BATCH, SEQ_LEN, d_model)
        *batch_size, seq_len, d_model = X.shape
        assert d_model % 2 == 0, "Error: the embedding dimension should be divisible by 2"

        rotation_mat = self.__build_rotation_matrix(seq_len, d_model, device=X.device) # (seq_len, d_model // 2, 2, 2)
        X_reshaped = X.view(*batch_size, seq_len, d_model // 2, 1, 2) # (...batch, seq_len, d_model // 2, 1, 2)

        return (
            (X_reshaped * rotation_mat) # (...batch, seq_len, d_model // 2, 2, 2)
            .sum(dim=-1) # (...batch, seq_len, d_model // 2, 2)
            .view(*batch_size, seq_len, d_model) # (...batch, seq_len, d_model)
        )
    
    def __build_rotation_matrix(self, seq_len, d_model, device):
        # try to retrieve from cache
        if (seq_len, d_model) in self.cache:
            return self.cache[(seq_len, d_model)]

        lengths = torch.arange(0, seq_len, requires_grad=False, device=device) # (seq_len)
        thetas_inds = torch.arange(0, d_model // 2, requires_grad=False, device=device) # (d_model // 2)
        thetas = torch.exp(-2 * math.log(10000) * (thetas_inds // 2) / d_model) # (d_model // 2)

        prod = torch.einsum("a , b -> ab", lengths, thetas) # (seq_len, d_model // 2)
        cosines = torch.cos(prod) # (seq_len, d_model // 2)
        sines = torch.sin(prod) # (seq_len, d_model // 2)

        rotation_mat = (
            torch
            .stack([cosines, -sines, sines, cosines], dim=-1)
            .view(seq_len, d_model // 2, 2, 2)
        ) # (seq_len, d_model // 2, 2, 2)

        # save to cache
        self.cache[(seq_len, d_model)] = rotation_mat

        return rotation_mat


class RotaryMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        bias=True,
        vdim=None,
    ):
        assert d_model % nhead == 0, "Error: the embedding dimension should be divisible by the number of heads"

        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.vdim = self.head_dim if vdim is None else vdim
        
        dim_qkv = d_model * 2 + self.vdim * nhead
        self.W_QKV = nn.Linear(d_model, dim_qkv, bias=bias)
        self.WO = nn.Linear(nhead * self.vdim, d_model, bias=bias)

        self.rotary_embedding = RotaryEmbedding()

    def forward(
        self,
        embeddings,  # (BATCH, SEQ_LEN, d_model)
        attention_mask=None,  # (BATCH, SEQ_LEN)
    ):  # -> (BATCH, TARGET_SEQ_DIM, d_model) ; OPT attention weights
        QKV = self.W_QKV(embeddings)  # (BATCH, SEQ_LEN, 2 * d_model + VDIM)
        Q, K, V = QKV.split(
            [self.d_model, self.d_model, self.vdim * self.nhead],
            dim=-1,
        )  # (BATCH, SEQ_LEN, d_model), (BATCH, SEQ_LEN, d_model), (BATCH, SEQ_LEN, VDIM)

        Q = self.__separate_heads(Q)  # (BATCH, nhead, SEQ_LEN, head_dim)
        K = self.__separate_heads(K)  # (BATCH, nhead, SEQ_LEN, head_dim)
        V = self.__separate_heads(V)  # (BATCH, nhead, SEQ_LEN, vdim)

        Q = self.rotary_embedding.apply(Q)  # (BATCH, nhead, SEQ_LEN, head_dim)
        K = self.rotary_embedding.apply(K)  # (BATCH, nhead, SEQ_LEN, head_dim)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (BATCH, 1, 1, SEQ_LEN)
        heads = F.scaled_dot_product_attention(Q, K, V, attn_mask=attention_mask)  # (...BATCH, nhead, SEQ_LEN, vdim)
        return self.WO(self.__join_heads(heads))

    def __separate_heads(self, mat):
        # (...BATCH, SEQ, nhead * proj_dim) -> (...BATCH, nhead, SEQ, proj_dim)
        *batch_dim, seq_dim, proj_dim = mat.shape
        return mat.view(*batch_dim, seq_dim, self.nhead, proj_dim // self.nhead).transpose(-2, -3)
    
    def __join_heads(self, mat):
        # (...BATCH, nhead, SEQ, proj_dim) -> (...BATCH, SEQ, nhead * proj_dim)
        *batch_dim, nhead, seq_dim, proj_dim = mat.shape
        return mat.transpose(-2, -3).contiguous().view(*batch_dim, seq_dim, nhead * proj_dim)


class RotaryTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=4,
        dropout=0.1,
        dim_feedforward=2048,
        activation_fn_cls=nn.ReLU,
        layer_norm_eps=1e-05,
    ):
        super().__init__()
        
        self.mh_attention = RotaryMultiheadSelfAttention(d_model, nhead)
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, layer_norm_eps)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation_fn_cls(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm_2 = nn.LayerNorm(d_model, layer_norm_eps)
    
    def forward(
        self,
        embeddings,  # (BATCH, LENGTH, d_model)
        attention_mask=None,  # (BATCH, LENGTH)
        token_type_ids=None,  # (BATCH, LENGTH)
    ):
        x = self.mh_attention(embeddings, attention_mask)
        x = self.dropout_1(x)
        x = self.layer_norm_1(embeddings + x)
        y = self.ff(x)
        return self.layer_norm_2(x + y)