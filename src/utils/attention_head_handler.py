import torch

class AttentionHeadHandler:
    @classmethod
    def separate_heads(cls, embeddings: torch.Tensor, nhead: int):
        # [B, L, D] -> [B, H, L, D // H]
        *B, L, D = embeddings.shape
        return embeddings.view(*B, L, nhead, D // nhead).transpose(-2, -3)
    
    @classmethod
    def join_heads(cls, embeddings):
        # [B, H, L, D // H] -> [B, L, D]
        *B, H, L, D = embeddings.shape
        return embeddings.transpose(-2, -3).contiguous().view(*B, L, H * D)