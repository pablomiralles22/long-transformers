import torch

from torch import nn

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