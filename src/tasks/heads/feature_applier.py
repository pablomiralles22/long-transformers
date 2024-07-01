import torch
import torch.nn as nn

from typing import Literal

Feature = Literal["diff", "dot"]

FEATURE_TO_FN = {
    "diff": lambda x, y: x - y,
    "dot": lambda x, y: x * y,
}

class FeatureApplier(nn.Module):
    def __init__(
        self,
        concat_consecutive: bool = False,
        features: tuple[Feature] = ("dot",),
    ):
        super(FeatureApplier, self).__init__()

        self.concat_consecutive = concat_consecutive
        self.features = features


    def forward(
        self,
        embeddings,  # [B, ..., D]
    ):
        if self.concat_consecutive is False:
            return embeddings
        
        x, y = embeddings[::2, ...], embeddings[1::2, ...]
        features = [FEATURE_TO_FN[feature](x, y) for feature in self.features]
        return torch.cat([x, y] + features, dim=-1)
    
    def get_output_dim(self, input_dim: int):
        if self.concat_consecutive is False:
            return input_dim
        return input_dim * (2 + len(self.features))