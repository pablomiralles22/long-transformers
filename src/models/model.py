from abc import ABC, abstractmethod
from torch import nn

class Model(nn.Module, ABC):
    def __init__(self):
        super(Model, self).__init__()

    @abstractmethod
    def forward(
        self,
        embeddings,  # (...BATCH, LENGTH, EMBED_DIM)
        attention_mask=None,  # (...BATCH, LENGTH)
        token_type_ids=None,  # (...BATCH, LENGTH)
    ):
        raise NotImplementedError

    @abstractmethod
    def get_input_embedding_dim(self):
        raise NotImplementedError

    @abstractmethod
    def get_output_embedding_dim(self):
        raise NotImplementedError
