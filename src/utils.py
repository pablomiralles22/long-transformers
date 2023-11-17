import torch
import math

from torch import nn


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

class ModelWithClassificationHead(nn.Module):
    def __init__(self, model, d_model):
        super(ModelWithClassificationHead, self).__init__()

        self.model = model
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        input_ids,  # (...BATCH, LENGTH)
        attention_mask=None,  # (...BATCH, LENGTH)
    ):
        x = self.model(input_ids, attention_mask)
        x = x[:, 0, :]
        x = self.head(x)
        return torch.sigmoid(x)