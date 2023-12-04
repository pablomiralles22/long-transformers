from torch import nn
from src.models.model import Model
from src.models.layers.layer import Layer

class LayeredModel(Model):
    def __init__(
        self,
        input_embedding_dim: int,
        output_embedding_dim: int,
        layers: list[Layer],
    ):
        super(LayeredModel, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.input_embedding_dim = input_embedding_dim
        self.output_embedding_dim = output_embedding_dim

    def forward(
        self,
        embeddings,  # (...BATCH, LENGTH, EMBED_DIM)
        attention_mask=None,  # (...BATCH, LENGTH)
        token_type_ids=None,  # (...BATCH, LENGTH)
    ):
        for layer in self.layers:
            embeddings = layer(embeddings, attention_mask, token_type_ids)
        return embeddings

    def get_input_embedding_dim(self):
        return self.input_embedding_dim

    def get_output_embedding_dim(self):
        return self.output_embedding_dim
