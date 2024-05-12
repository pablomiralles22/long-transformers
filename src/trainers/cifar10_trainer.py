import torch
import torchmetrics
import pytorch_lightning as pl

from copy import deepcopy
from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from src.data_loaders.data_module_builder import DataModuleBuilder
from src.models.model_builder import ModelBuilder
from src.heads.classification_head import get_model_with_classification_head
from src.utils.l1_regularizer import L1Regularizer
from src.utils.weight_decay_param_filter import WeightDecayParamFilter

class CIFAR10Module(pl.LightningModule):
    @classmethod
    def get_default_optimizer_config(cls) -> dict:
        return {
            "lr": 1e-4,
            "betas": (0.9, 0.99),
            "weight_decay": 0.0,
            "l1_lambda": 0.0,
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
        self.l1_lambda = self.optimizer_config.pop("l1_lambda")

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
        self.log_dict({ "tr_loss_step": loss, "tr_acc_step": accuracy, }, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log_dict({ "tr_loss": loss, "tr_acc": accuracy, }, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        # learning_rate = self.lr_schedulers().get_last_lr()[0]
        # self.log("lr", learning_rate, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        return {"loss": loss, "logits": logits, "labels": labels}

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._step(batch, batch_idx)
        accuracy = self.accuracy(logits, labels)
        self.log_dict({ "val_loss": loss, "val_acc": accuracy, }, on_step=False, on_epoch=True, prog_bar=True,)
        return {"loss": loss, "logits": logits, "labels": labels}

    def _step(self, batch, _):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self.model_with_head(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
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

        L1Regularizer.apply(self.model_with_head, l1_lambda=self.l1_lambda)

        # set up scheduler
        train_len = len(self.data_module.train_dataloader())
        max_epochs = self.trainer.max_epochs
        swap_point = int(0.25 * max_epochs * train_len)

        linear_lr = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=1., total_iters=swap_point)
        cosine_anneal_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
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
                patience=20,
                mode="max",
            ),
        ]

