import torch

from torch import nn

class ModelWithClassificationHead(nn.Module):
    def __init__(self, model, d_model, reduction="first_token"):
        super(ModelWithClassificationHead, self).__init__()

        self.model = model
        self.head = nn.Linear(d_model, 1)
        self.reduction = reduction

    def forward(
        self,
        input_ids,  # (...BATCH, LENGTH)
        attention_mask=None,  # (...BATCH, LENGTH)
    ):
        x = self.model(input_ids, attention_mask)
        match self.reduction:
            case "first_token":
                x = x[..., 0, :]
            case "mean":
                x = x.mean(dim=-2)
            case _:
                raise ValueError(f"Invalid reduction: {self.reduction}")
        x = self.head(x)
        return torch.sigmoid(x)