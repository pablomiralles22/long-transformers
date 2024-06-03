import torch
import torchmetrics
import pytorch_lightning as pl

from copy import deepcopy
from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from src.data_loaders.data_module_builder import DataModuleBuilder
from src.models.model_builder import ModelBuilder
from src.heads.classification_head import ClassificationHead
from src.utils.l1_regularizer import L1Regularizer
from src.utils.weight_decay_param_filter import WeightDecayParamFilter

class TextClassificationModule(pl.LightningModule):
    @classmethod
    def get_default_optimizer_config(cls) -> dict:
        return {
            "lr": 1e-4,
            "betas": (0.9, 0.99),
            "weight_decay": 0.0,
            "l1_lambda": 0.0,
            "mlm_aux_task": False,
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
        self.mlm_aux_task = self.optimizer_config.pop("mlm_aux_task")

        if self.mlm_aux_task is False:
            data_module_params["params"]["mask_ratio"] = 0.0

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
        self.head = ClassificationHead(
            input_dim=self.model.get_output_embedding_dim(),
            **head_params,
        )
        # auxiliar task
        head_params["reduction_method"] = "none"
        head_params["num_hidden_layers"] = 0
        head_params["num_classes"] = self.data_module.get_vocab_size()
        head_params["concat_consecutive"] = False
        self.head_aux_task = ClassificationHead(
            input_dim=self.model.get_output_embedding_dim(),
            **head_params,
        )

        # create metrics
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def training_step(self, batch, batch_idx):
        loss, loss_1, loss_2, logits_1, labels = self._step(batch, batch_idx)
        accuracy = self.accuracy(logits_1, labels)
        self.log_dict(
            { "train_loss_1": loss_1, "train_loss_2": loss_2, "train_accuracy": accuracy },
            on_step=False, on_epoch=True, prog_bar=False, logger=True,
        )
        self.log_dict(
            { "train_loss_1_step": loss_1, "train_loss_2_step": loss_2 },
            on_step=True, on_epoch=False, prog_bar=True, logger=False,
        )
        return loss if self.mlm_aux_task else loss_1

    def validation_step(self, batch, batch_idx):
        loss, loss_1, loss_2, logits_1, labels = self._step(batch, batch_idx)
        accuracy = self.accuracy(logits_1, labels)
        self.log_dict(
            { "val_loss_1": loss_1, "val_loss_2": loss_2, "val_accuracy": accuracy, },
            on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )
        return loss if self.mlm_aux_task else loss_1

    def _step(self, batch, _):
        input_ids = batch["input_ids"]
        corrupted_input_ids = batch["corrupted_input_ids"]  # [B, L]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].float().reshape(-1, 1)

        x = self.model(corrupted_input_ids, attention_mask)  # [B, L, D]
        B, L, D = x.shape

        logits_1 = self.head(x, attention_mask)  # [B, 1]
        logits_2 = self.head_aux_task(x, attention_mask)  # [B, L, V]

        loss_1 = F.binary_cross_entropy_with_logits(logits_1, labels)

        token_loss_2 = F.cross_entropy(
            logits_2.view(B * L, -1),
            input_ids.view(B * L),
            reduction="none"
        )  # [B * L]
        loss_2 = (token_loss_2 * attention_mask.view(-1)).sum() / attention_mask.sum()

        loss = loss_1 + loss_2

        return loss, loss_1, loss_2, logits_1, labels

    def configure_optimizers(self):
        # set up optimizer
        weight_decay_params, no_weight_decay_params = WeightDecayParamFilter.filter(self)

        general_optimizer_config = deepcopy(self.optimizer_config)
        weight_decay = general_optimizer_config.pop("weight_decay")
        general_optimizer_config["weight_decay"] = 0.0

        optim_groups = [
            {"params": weight_decay_params, "weight_decay": weight_decay},
            {"params": no_weight_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, **general_optimizer_config)

        if self.l1_lambda > 0.:
            L1Regularizer.apply(self, l1_lambda=self.l1_lambda)

        # set up scheduler
        train_len = len(self.data_module.train_dataloader())
        max_epochs = self.trainer.max_epochs
        swap_point = int(0.5 * max_epochs * train_len)

        linear_lr = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=1., total_iters=swap_point)
        cosine_anneal_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
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
                filename="{epoch}-{val_accuracy:.2f}",
                monitor="val_accuracy",
                mode="max",
            ),
            EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                mode="max",
            ),
        ]

