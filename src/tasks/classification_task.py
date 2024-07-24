import torch
import torch.nn as nn

from torchmetrics import Accuracy
from src.tasks.task import Task
from src.tasks.heads.classification_head import ClassificationHead


class ClassificationTask(Task):
    def __init__(self, num_classes: int, head_params: dict):
        super().__init__()
        self.num_classes = num_classes

        if num_classes == 2:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        head_params["num_classes"] = num_classes
        self.head = ClassificationHead(**head_params)

        task = "binary" if num_classes == 2 else "multiclass"
        self.accuracy = Accuracy(task=task, num_classes=num_classes)
        

    def forward(self, inputs: dict, outputs: torch.Tensor) -> dict:
        # inputs:
        #   input_ids: [B, L]
        #   attention_mask: [B, L]
        #   labels: [B]
        # outputs: [B, L, D]
        logits = self.head(outputs, inputs["attention_mask"])  # [B, num_classes]
        labels = inputs["labels"]

        if self.num_classes == 2:
            labels = labels.float().reshape(-1, 1)

        loss = self.loss_fn(logits, labels)
        acc = self.accuracy(logits, labels)

        return {"loss": loss, "accuracy": acc}

    def get_metric_to_track(self) -> tuple[str, str]:
        return "classification_accuracy", "max"