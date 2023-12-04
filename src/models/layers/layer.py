from abc import ABC, abstractmethod
from torch import nn

class Layer(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        embeddings,  # (BATCH, LENGTH, EMBED_DIM)
        attention_mask=None,  # (BATCH, LENGTH)
        token_type_ids=None,  # (BATCH, LENGTH)
    ):
        raise NotImplementedError("This is an abstract class")
    

class Overlayer(Layer):
    def __init__(self, layer: nn.Module):
        super(Overlayer, self).__init__()
        self.layer = layer
    
    def forward(
        self,
        embeddings,  # (BATCH, LENGTH, EMBED_DIM)
        attention_mask=None,  # (BATCH, LENGTH)
        token_type_ids=None,  # (BATCH, LENGTH)
    ):
        return self.layer(embeddings)

class ConvLayer(Layer):
    def __init__(self, layer: nn.Module):
        super(ConvLayer, self).__init__()
        self.layer = layer
    
    def forward(
        self,
        embeddings,  # (BATCH, LENGTH, EMBED_DIM)
        attention_mask=None,  # (BATCH, LENGTH)
        token_type_ids=None,  # (BATCH, LENGTH)
    ):
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings * attention_mask.unsqueeze(1)  # set padding to 0
        embeddings = self.layer(embeddings)
        embeddings = embeddings.transpose(1, 2)
        return embeddings
