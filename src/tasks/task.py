import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class Task(nn.Module, ABC):
    @abstractmethod
    def forward(self, inputs: dict, outputs: torch.Tensor) -> dict:
        pass

    @abstractmethod
    def get_metric_to_track(self) -> tuple[str, str]:
        pass

    def preprocess_input(self, inputs: dict, train: bool = False) -> dict:
        return inputs
