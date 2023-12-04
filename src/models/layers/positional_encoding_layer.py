import torch
import math

from src.models.layers.layer import Layer


def positional_encoding(length, d_model, device=None):
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            f"odd dim (got dim={d_model})"
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    pe.requires_grad = False

    return pe.to(device)

class PositionalEncodingLayer(Layer):
    def forward(
        self,
        embeddings,  # (BATCH, LENGTH, EMBED_DIM)
        attention_mask=None,  # (BATCH, LENGTH)
        token_type_ids=None,  # (BATCH, LENGTH)
    ):
        _, seq_len, embed_dim = embeddings.shape
        device = embeddings.device
        pe = positional_encoding(seq_len, embed_dim, device=device)
        return embeddings + pe
