import torch

from torch import nn
from src.models.layers.layer import Layer

class LearnedPositionalEncodingLayer(Layer):
    def __init__(self, max_seq_len: int, embed_dim: int):
        super(LearnedPositionalEncodingLayer, self).__init__()
        self.pe = nn.Embedding(max_seq_len, embed_dim)
    
    def forward(
        self,
        embeddings,  # (BATCH, LENGTH, EMBED_DIM)
        attention_mask=None,  # (BATCH, LENGTH)
        token_type_ids=None,  # (BATCH, LENGTH)
    ):
        _, seq_len, _ = embeddings.shape
        device = embeddings.device
        positions = torch.arange(seq_len, device=device).expand(embeddings.shape[:2])
        embeddings = embeddings + self.pe(positions)
        return embeddings 
