from torch import nn

from src.models.layers.layer import Layer

class ResidualBlock(Layer):
    def __init__(
        self,
        layers: list[Layer],
    ):
        super(ResidualBlock, self).__init__()
        self.sublayers = nn.ModuleList(layers)

    def forward(
        self,
        embeddings,  # (BATCH, LENGTH, EMBED_DIM)
        attention_mask=None,  # (BATCH, LENGTH)
        token_type_ids=None,  # (BATCH, LENGTH)
    ):
        embeddings_skip = embeddings
        for layer in self.sublayers:
            embeddings = layer.forward(embeddings, attention_mask, token_type_ids)
        return embeddings_skip + embeddings
