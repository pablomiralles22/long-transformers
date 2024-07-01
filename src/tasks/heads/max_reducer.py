import torch

class MaxReducer:
    @classmethod
    def reduce(self, x, attention_mask):
        # x: [B, L, D]
        # attention_mask: [B, L]

        if x.size(1) != attention_mask.size(1):
            return torch.max(x, dim=-2)
        
        attention_mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
        x = x.masked_fill(attention_mask == 0, float('-inf'))
        return torch.max(x, dim=-2).values

    @classmethod
    def soft_reduce(self, x, attention_mask):
        # x: [B, L, D]
        # attention_mask: [B, L]

        if x.size(1) != attention_mask.size(1):
            return torch.max(x, dim=-2)
        
        attention_mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
        pre_softmax = x.masked_fill(attention_mask == 0, float('-inf'))
        return (pre_softmax.softmax(dim=-2) * x).sum(dim=-2)