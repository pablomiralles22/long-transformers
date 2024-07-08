import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.task import Task
from src.tasks.heads.classification_head import ClassificationHead


class DenoiseTask(Task):
    def __init__(
        self,
        num_embeddings: int,
        mask_token_id: int,
        mask_ratio: float,
        random_ratio: float,
        head_params: dict,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.mask_token_id = mask_token_id
        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio

        if num_embeddings == 2:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        head_params["num_classes"] = num_embeddings
        self.head = ClassificationHead(**head_params)
        

    def forward(self, inputs: dict, outputs: torch.Tensor) -> dict:
        # inputs:
        #   input_ids: [B, L]
        #   uncorrupted_input_ids: [B, L]
        #   attention_mask: [B, L]
        #   labels: [B]
        # outputs: [B, L, D]
        B, L, _ = outputs.shape
        V = self.num_embeddings

        flat_labels = inputs["uncorrupted_input_ids"].view(-1)
        flat_attn_mask = inputs["attention_mask"].view(-1)

        logits = self.head(outputs, inputs["attention_mask"])  # [B, L, V]

        tokenwise_loss = F.cross_entropy(logits.view(B * L, V), flat_labels, reduction="none")  # [B * L]
        loss = torch.mean(tokenwise_loss[flat_attn_mask])

        return {"loss": loss}

    def get_metric_to_track(self) -> tuple[str, str]:
        return "denoise_loss", "min"

    def preprocess_input(self, inputs: dict, train: bool = False) -> dict:
        # inputs:
        #   input_ids: [B, L]
        #   attention_mask: [B, L]
        #   labels: [B]
        if train is False:
            inputs["uncorrupted_input_ids"] = inputs["input_ids"]
            return inputs

        # corrupt data
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        device = input_ids.device

        mask_indices = torch.rand(input_ids.shape, device=device) < self.mask_ratio
        mask_indices &= attention_mask.bool()
        
        random_indices = torch.rand(input_ids.shape, device=device) < self.random_ratio
        random_indices &= mask_indices

        corrupted_input_ids = input_ids.clone()
        corrupted_input_ids[mask_indices] = self.mask_token_id
        corrupted_input_ids[random_indices] = torch.randint(
            0, self.num_embeddings, torch.where(random_indices)[0].shape, device=device
        )

        # modify batch
        inputs["input_ids"] = corrupted_input_ids
        inputs["uncorrupted_input_ids"] = input_ids

        return inputs

        