from torch import nn
from src.models.model_with_embedding import ModelWithEmbedding
from src.custom_types import ReductionMethod


def get_model_with_classification_head(
    model: ModelWithEmbedding,
    ff_dim: int,
    dropout_p: float = 0.1,
    num_hidden_layers: int = 1,
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
        dropout_p=dropout_p,
        reduction_method=reduction_method,
    )


def linear_block(input_dim: int, output_dim: int, dropout_p: float = 0.1):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.Dropout(dropout_p),
        nn.ReLU(),
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
        dropout_p: float = 0.1,
        reduction_method: ReductionMethod = "cls",
    ):
        super(ModelWithClassificationHead, self).__init__()
        self.model = model
        self.ff = nn.Sequential(
            linear_block(input_dim, ff_dim, dropout_p),
            *[
                linear_block(ff_dim, ff_dim, dropout_p)
                for _ in range(num_hidden_layers - 1)
            ],
            nn.Linear(ff_dim, 1),
        )
        self.reduction_method = reduction_method

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The output logits.
        """
        x = self.model(*args, **kwargs)
        logits = self.ff(self._reduce(x))
        return logits

    def _reduce(self, x):
        if self.reduction_method == "cls":
            return x[:, 0, :]
        if self.reduction_method == "mean":
            return x.mean(dim=1)
        raise ValueError(f"Unknown reduction method: {self.reduction_method}")


