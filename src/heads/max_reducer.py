import torch

class MaxReducer:
    @classmethod
    def reduce(self, x, attention_mask):
        # x: [B, L, D]
        # attention_mask: [B, L]

        if x.size(1) != attention_mask.size(1):
            return torch.max(x, dim=-2)
        
        attention_mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
        x.masked_fill_(attention_mask == 0, float('-inf'))
        return torch.max(x, dim=-2).values