from typing import Literal

ReductionMethod = Literal[
    "cls",
    "mean",
    "max",
    "softmax",
    "sum",
    "attention",
    "glu",
    "none",
]