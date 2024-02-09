import torch
import torchmetrics
import pytorch_lightning as pl

from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from src.data_loaders.data_module_builder import DataModuleBuilder
from src.models.model_builder import ModelBuilder
from src.heads.classification_head import get_model_with_classification_head

class CIFAR10Module(pl.LightningModule):
    @classmethod
    def get_default_optimizer_config(cls) -> dict:
        return {
            "lr": 1e-4,
            "betas": (0.9, 0.99),
            "weight_decay": 0.0,
        }

    def __init__(
        self,
        model_params: dict,
        data_module_params: dict,
        head_params: dict,
        optimizer_params: dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        # optimizer config
        self.optimizer_config = {
            **self.get_default_optimizer_config(),
            **optimizer_params,
        }

        # build data module
        self.data_module = DataModuleBuilder.build_data_module(
            **data_module_params
        )
        # build model
        self.model = ModelBuilder.build_model(
            **model_params,
            vocab_size=self.data_module.get_vocab_size(),
            padding_idx=self.data_module.get_pad_token_id(),
        )
        self.model_with_head = get_model_with_classification_head(
            model=self.model,
            **head_params,
        )

        # create metrics
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._step(batch, batch_idx)
        accuracy = self.accuracy(logits, labels)
        self.log_dict(
            {
                "tr_loss": loss,
                "tr_acc": accuracy,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "logits": logits, "labels": labels}

    def validation_step(self, batch, batch_idx):
        # self.model.train() # dirty fix for pytorch bug
        loss, logits, labels = self._step(batch, batch_idx)
        accuracy = self.accuracy(logits, labels)
        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "logits": logits, "labels": labels}

    def _step(self, batch, _):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self.model_with_head(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        return loss, logits, labels

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(), **self.optimizer_config
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.25, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "tr_acc",
                "interval": "epoch",
            },
        }
    
    def configure_callbacks(self) -> list[Callback]:
        return [
            ModelCheckpoint(
                filename="{epoch}-{val_acc:.2f}",
                monitor="val_acc",
                mode="max",
            ),
            EarlyStopping(
                monitor="val_acc",
                patience=10,
                mode="max",
            ),
        ]

