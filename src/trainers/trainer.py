import torch
import torchmetrics
import pytorch_lightning as pl

from torch.nn import functional as F

class TextClassificationModule(pl.LightningModule):
    @classmethod
    def get_default_optimizer_config(cls) -> dict:
        return {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
        }

    def __init__(
        self,
        model,
        optimizer_config,
    ):
        super().__init__()
        self.optimizer_config = {
            **self.get_default_optimizer_config(),
            **optimizer_config,
        }
        self.model = model
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._step(batch, batch_idx)
        accuracy = self.accuracy(logits, labels)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "logits": logits, "labels": labels}

    def validation_step(self, batch, batch_idx):
        self.model.train() # dirty fix for pytorch bug
        loss, logits, labels = self._step(batch, batch_idx)
        accuracy = self.accuracy(logits, labels)
        self.log_dict(
            {
                "val_loss": loss,
                "val_accuracy": accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "logits": logits, "labels": labels}

    def _step(self, batch, _):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].float().reshape(-1, 1)

        logits = self.model(input_ids, attention_mask)
        loss = F.binary_cross_entropy(logits, labels)
        return loss, logits, labels

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), **self.optimizer_config
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_accuracy",
                "interval": "epoch",
            },
        }
