import torch

from typing import Optional, Literal
from torch import nn
from src.models.layers.layer import Layer
from src.utils.sinusoidal_positional_embedding import sinusoidal_positional_embedding
from src.utils.hyperbolic_positional_embedding import hyperbolic_positional_embedding
from src.utils.log_positional_embedding import log_positional_embedding


class PositionalEmbeddingLayer(Layer):
    def __init__(
        self,
        mode: Literal["sinusoidal", "hyperbolic", "log", "learned"],
        max_seq_len: Optional[int] = None,
        embed_dim: Optional[int] = None,
    ):
        super(PositionalEmbeddingLayer, self).__init__()
        assert mode in ["sinusoidal", "hyperbolic", "log"] or (
            max_seq_len is not None and embed_dim is not None
        ), "max_seq_len and embed_dim must be provided for learned positional embeddings"

        self.mode = mode
        if mode == "learned":
            self.pe = nn.Embedding(max_seq_len, embed_dim)

    def forward(
        self,
        embeddings,  # (BATCH, LENGTH, EMBED_DIM)
        attention_mask=None,  # (BATCH, LENGTH)
        token_type_ids=None,  # (BATCH, LENGTH)
    ):
        if self.mode == "learned":
            batch_dim, seq_len, _ = embeddings.shape
            device = embeddings.device
            positions = (
                torch
                .arange(seq_len, device=device)
                .expand(batch_dim, seq_len)
            )
            embeddings = embeddings + self.pe(positions)
            return embeddings
        elif self.mode == "sinusoidal":
            _, seq_len, embed_dim = embeddings.shape
            dtype, device = embeddings.dtype, embeddings.device
            idxs = torch.arange(seq_len, dtype=dtype, device=device)
            pe = sinusoidal_positional_embedding(idxs, embed_dim, device=device)
            return embeddings + pe
        elif self.mode == "hyperbolic":
            _, seq_len, embed_dim = embeddings.shape
            dtype, device = embeddings.dtype, embeddings.device
            idxs = torch.arange(seq_len, dtype=dtype, device=device)
            pe = hyperbolic_positional_embedding(idxs, embed_dim, device=device)
            return embeddings + pe
        elif self.mode == "log":
            _, seq_len, embed_dim = embeddings.shape
            dtype, device = embeddings.dtype, embeddings.device
            idxs = torch.arange(seq_len, dtype=dtype, device=device)
            pe = log_positional_embedding(idxs, embed_dim, device=device)
            return embeddings + pe

class TokenTypeEmbeddingLayer(Layer):
    def __init__(
        self,
        mode: Literal["sinusoidal", "learned"],
        max_seq_len: Optional[int] = None,
        embed_dim: Optional[int] = None,
    ):
        super(TokenTypeEmbeddingLayer, self).__init__()
        assert type == "sinusoidal" or (
            max_seq_len is not None and embed_dim is not None
        ), "max_seq_len and embed_dim must be provided for learned positional embeddings"

        self.mode = mode
        if mode == "learned":
            self.pe = nn.Embedding(max_seq_len, embed_dim)

    def forward(
        self,
        embeddings,  # [B, L, D]
        attention_mask=None,  # [B, L]
        token_type_ids=None,  # [B, L]
    ):
        if self.mode == "learned":
            embeddings = embeddings + self.pe(token_type_ids)
            return embeddings
        elif self.mode == "sinusoidal":
            _, _, embed_dim = embeddings.shape
            device = embeddings.device
            pe = sinusoidal_positional_embedding(token_type_ids, embed_dim, device=device)
            return embeddings + pe
