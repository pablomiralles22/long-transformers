from torch import nn
from src.models.model import Model

class ModelWithEmbedding(nn.Module):
    def __init__(
        self,
        model: Model,
        vocab_size: int,
        padding_idx: int,
    ):
        super(ModelWithEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            model.get_input_embedding_dim(),
            padding_idx=padding_idx,
        )
        self.model = model

    def forward(
        self,
        input_ids,  # (BATCH, LENGTH)
        attention_mask=None,  # (BATCH, LENGTH)
        token_type_ids=None,  # (BATCH, LENGTH)
    ):
        embeddings = self.embedding(input_ids)
        return self.model(embeddings, attention_mask, token_type_ids)

    def get_output_embedding_dim(self):
        return self.model.get_output_embedding_dim()
