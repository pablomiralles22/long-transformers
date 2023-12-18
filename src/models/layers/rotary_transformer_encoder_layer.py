import torch
import math

from torch import nn
from torch.nn import functional as F
from src.models.layers.layer import Layer


class RotaryEmbedding:
    __CACHE = dict()

    @classmethod
    def apply(cls, X, freq=10000):
        # X: (BATCH, SEQ_LEN, d_model)
        *batch_size, seq_len, d_model = X.shape
        assert d_model % 2 == 0, "Error: the embedding dimension should be divisible by 2"

        device, dtype = X.device, X.dtype

        rotation_mat = cls.__build_rotation_matrix(seq_len, d_model, freq, device=device, dtype=dtype) # (seq_len, d_model // 2, 2, 2)
        X_reshaped = X.view(*batch_size, seq_len, d_model // 2, 1, 2) # (...batch, seq_len, d_model // 2, 1, 2)

        return (
            (X_reshaped * rotation_mat) # (...batch, seq_len, d_model // 2, 2, 2)
            .sum(dim=-1) # (...batch, seq_len, d_model // 2, 2)
            .view(*batch_size, seq_len, d_model) # (...batch, seq_len, d_model)
        )
    
    @classmethod
    def __build_rotation_matrix(cls, seq_len, d_model, freq, device, dtype):
        # try to retrieve from cache
        if (seq_len, d_model, freq) in cls.__CACHE:
            return cls.__CACHE[(seq_len, d_model, freq)]

        lengths = torch.arange(0, seq_len, requires_grad=False, device=device, dtype=dtype) # (seq_len)
        thetas_inds = torch.arange(0, d_model // 2, requires_grad=False, device=device, dtype=dtype) # (d_model // 2)
        thetas = torch.exp(-2 * math.log(freq) * (thetas_inds // 2) / d_model) # (d_model // 2)

        prod = torch.einsum("a , b -> ab", lengths, thetas) # (seq_len, d_model // 2)
        cosines = torch.cos(prod) # (seq_len, d_model // 2)
        sines = torch.sin(prod) # (seq_len, d_model // 2)

        rotation_mat = (
            torch
            .stack([cosines, -sines, sines, cosines], dim=-1)
            .view(seq_len, d_model // 2, 2, 2)
        ) # (seq_len, d_model // 2, 2, 2)

        # save to cache
        cls.__CACHE[(seq_len, d_model, freq)] = rotation_mat

        return rotation_mat


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
        embeddings,  # (BATCH, SEQ_LEN, d_model)
        attention_mask=None,  # (BATCH, SEQ_LEN)
    ):  # -> (BATCH, SEQ_LEN, d_model)
        QKV = self.W_QKV(embeddings)  # (BATCH, SEQ_LEN, 2 * qkdim + vdim)
        Q, K, V = QKV.split(
            [self.qkdim, self.qkdim, self.vdim],
            dim=-1,
        )  # (BATCH, SEQ_LEN, qkdim), (BATCH, SEQ_LEN, qkdim), (BATCH, SEQ_LEN, vdim)

        Q = self.__separate_heads(Q)  # (BATCH, nhead, SEQ_LEN, qkdim)
        K = self.__separate_heads(K)  # (BATCH, nhead, SEQ_LEN, qkdim)
        V = self.__separate_heads(V)  # (BATCH, nhead, SEQ_LEN, vdim)

        Q = RotaryEmbedding.apply(Q, freq=self.freq)  # (BATCH, nhead, SEQ_LEN, qkdim)
        K = RotaryEmbedding.apply(K, freq=self.freq)  # (BATCH, nhead, SEQ_LEN, qkdim)
        
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


class RotaryTransformerEncoderLayer(Layer):
    def __init__(
        self,
        d_model, vdim=None, qkdim=None,
        nhead=4,
        dropout=0.1,
        dim_feedforward=2048,
        activation_fn_cls=nn.ReLU,
        layer_norm_eps=1e-05,
        freq=10000,
        skip_connection=True,
    ):
        super().__init__()
        
        self.mh_attention = RotaryMultiheadSelfAttention(d_model, nhead, vdim=vdim, qkdim=qkdim, freq=freq)
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, layer_norm_eps)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation_fn_cls(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm_2 = nn.LayerNorm(d_model, layer_norm_eps)

        self.skip_connection = skip_connection
    
    def forward(
        self,
        embeddings,  # (BATCH, LENGTH, d_model)
        attention_mask=None,  # (BATCH, LENGTH)
        token_type_ids=None,  # (BATCH, LENGTH)
    ):
        if self.skip_connection is True:
            x = self.mh_attention(embeddings, attention_mask)
            x = self.dropout_1(x)
            x = self.layer_norm_1(embeddings + x)
            y = self.ff(x)
            return self.layer_norm_2(x + y)
        
        x = self.mh_attention(embeddings, attention_mask)
        x = self.dropout_1(x)
        x = self.layer_norm_1(x)
        x = self.ff(x)
        return self.layer_norm_2(x)