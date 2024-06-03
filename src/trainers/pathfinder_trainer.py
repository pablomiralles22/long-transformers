from copy import deepcopy
import torch
import torchmetrics
import pytorch_lightning as pl

from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from src.data_loaders.data_module_builder import DataModuleBuilder
from src.models.model_builder import ModelBuilder
from src.heads.classification_head import get_model_with_classification_head
from src.utils.weight_decay_param_filter import WeightDecayParamFilter

class PathfinderModule(pl.LightningModule):
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
        self.accuracy = torchmetrics.Accuracy(task="binary")

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
        labels = batch["labels"].float().reshape(-1, 1)

        logits = self.model_with_head(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss, logits, labels

    def configure_optimizers(self):
        # set up optimizer
        weight_decay_params, no_weight_decay_params = WeightDecayParamFilter.filter(self.model_with_head)

        general_optimizer_config = deepcopy(self.optimizer_config)
        weight_decay = general_optimizer_config.pop("weight_decay")
        general_optimizer_config["weight_decay"] = 0.0

        optim_groups = [
            {"params": weight_decay_params, "weight_decay": weight_decay},
            {"params": no_weight_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, **general_optimizer_config)

        # set up scheduler
        train_len = len(self.data_module.train_dataloader())
        max_epochs = self.trainer.max_epochs
        swap_point = int(0.1 * max_epochs * train_len)

        linear_lr = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, end_factor=1., total_iters=swap_point)
        cosine_anneal_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [linear_lr, cosine_anneal_lr],
            [swap_point],
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
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

