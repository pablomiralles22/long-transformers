import torch.nn as nn

from typing import Literal

ActivationFn = Literal[
    "relu",
    "gelu",
    "tanh",
    "sigmoid",
    "hardsigmoid",
    "silu",
    "selu",
    "mish",
    "softmax",
]

def build_activation(fn: ActivationFn):
    match fn:
        case "relu":
            return nn.ReLU()
        case "gelu":
            return nn.GELU()
        case "tanh":
            return nn.Tanh()
        case "sigmoid":
            return nn.Sigmoid()
        case "softmax":
            return nn.Softmax(dim=-1)
        case "hardsigmoid":
            return nn.Hardsigmoid()
        case "silu":
            return nn.SiLU()
        case "selu":
            return nn.SELU()
        case "mish":
            return nn.Mish()
        case _:
            raise NotImplementedError(f"Activation function {fn} not implemented")