from torch import nn
from src.models.model_with_embedding import ModelWithEmbedding
from src.custom_types import ReductionMethod
from src.heads.attention_reducer import AttentionReducer
from src.heads.mean_reducer import MeanReducer
from src.heads.max_reducer import MaxReducer
from src.heads.glu_reducer import GLUReducer


def get_model_with_classification_head(
    model: ModelWithEmbedding,
    ff_dim: int,
    dropout_p: float = 0.1,
    num_hidden_layers: int = 1,
    num_classes: int = 2,
    reduction_method: ReductionMethod = "cls",
):
    """
    Builds a model with a classification head on top of the base model.
    For this, it uses the output embedding dimension of the base model,
    which is given by the `get_out_embedding_dim` method.
    """
    input_dim = model.get_output_embedding_dim()
    return ModelWithClassificationHead(
        model=model,
        input_dim=input_dim,
        ff_dim=ff_dim,
        num_hidden_layers=num_hidden_layers,
        num_classes=num_classes,
        dropout_p=dropout_p,
        reduction_method=reduction_method,
    )


def linear_block(input_dim: int, output_dim: int, dropout_p: float = 0.1):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.Dropout(dropout_p),
        nn.GELU(),
    )


class ModelWithClassificationHead(nn.Module):
    """
    A PyTorch module that combines a base model with a classification head.

    Args:
        model (nn.Module): The base model.
        input_dim (int): The input dimension of the classification head.
        ff_dim (int): The hidden dimension of the feed-forward layers in the classification head.
        num_hidden_layers (int, optional): The number of hidden layers in the classification head. Defaults to 1.
        dropout_p (float, optional): The dropout probability for the feed-forward layers. Defaults to 0.1.
    """

    def __init__(
        self,
        model: nn.Module,
        input_dim: int,
        ff_dim: int,
        num_hidden_layers: int = 1,
        num_classes: int = 2,
        dropout_p: float = 0.1,
        reduction_method: ReductionMethod = "cls",
    ):
        super(ModelWithClassificationHead, self).__init__()
        self.model = model
        output_dim = 1 if num_classes == 2 else num_classes
        self.ff = nn.Sequential(
            linear_block(input_dim, ff_dim, dropout_p),
            *[
                linear_block(ff_dim, ff_dim, dropout_p)
                for _ in range(num_hidden_layers - 1)
            ],
            nn.Linear(ff_dim, output_dim),
        )
        self.reduction_method = reduction_method
        if reduction_method == "attention":
            self.reducer = AttentionReducer(input_dim, dropout_p=dropout_p)
        elif reduction_method == "glu":
            self.reducer = GLUReducer(input_dim, dropout_p=dropout_p)

    def forward(
        self,
        embeddings,  # [B, L, D]
        attention_mask=None,  # [B, L]
        token_type_ids=None,  # [B, L]
    ):
        """
        Forward pass of the model.
        Returns:
            torch.Tensor: The output logits.
        """
        x = self.model(embeddings, attention_mask, token_type_ids)
        logits = self.ff(self._reduce(x, attention_mask))
        return logits

    def _reduce(self, x, attention_mask):
        if self.reduction_method == "cls":
            return x[:, 0, :]
        elif self.reduction_method == "mean":
            return MeanReducer.reduce(x, attention_mask)
        elif self.reduction_method == "max":
            return MaxReducer.reduce(x, attention_mask)
        elif self.reduction_method == "softmax":
            return MaxReducer.soft_reduce(x, attention_mask)
        elif self.reduction_method == "attention":
            return self.reducer(x, attention_mask)
        elif self.reduction_method == "glu":
            return MeanReducer.reduce(self.reducer(x, attention_mask), attention_mask)
        raise ValueError(f"Unknown reduction method: {self.reduction_method}")


