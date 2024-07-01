import torch

class MeanReducer:
    @classmethod
    def reduce(self, x, attention_mask):
        # x: [B, L, D]
        # attention_mask: [B, L]

        if x.size(1) != attention_mask.size(1):
            return torch.mean(x, dim=1)
        
        attention_mask = attention_mask.unsqueeze(-1).bool()  # [B, L, 1]
        x = x.masked_fill(~attention_mask, 0.)
        return (x / attention_mask.sum(1, keepdim=True)).sum(1)