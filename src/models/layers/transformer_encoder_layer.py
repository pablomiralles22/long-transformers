from torch import nn
from src.models.layers.layer import Layer


class TransformerEncoderLayer(Layer):
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderLayer, self).__init__()
        self.layer = nn.TransformerEncoderLayer(*args, **kwargs)

    def forward(
        self,
        embeddings,  # (BATCH, LENGTH, EMBED_DIM)
        attention_mask=None,  # (BATCH, LENGTH)
        token_type_ids=None,  # (BATCH, LENGTH)
    ):
        return self.layer(
            embeddings,
            src_key_padding_mask=attention_mask,
        )
