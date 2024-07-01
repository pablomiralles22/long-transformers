import pytorch_lightning as pl

from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from src.data_loaders.data_module_builder import DataModuleBuilder
from src.models.model_builder import ModelBuilder
from src.tasks.task_builder import TaskBuilder
from src.utils.optimizer_builder import OptimizerBuilder
from src.utils.scheduler_builder import SchedulerBuilder

class Trainer(pl.LightningModule):
    def __init__(
        self,
        model_params: dict,
        data_module_params: dict,
        tasks_params: list[dict],
        optimizer_params: dict,
        scheduler_params: dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        # optimizer config
        self.optimizer_config = optimizer_params

        # scheduler config
        self.scheduler_config = scheduler_params

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

        # build tasks
        self.metric_to_track = None
        self.tasks = dict()

        for task_params in tasks_params:
            task_name = task_params["task_name"]
            task_params["head_params"]["input_dim"] = self.model.get_output_embedding_dim()

            task = TaskBuilder.build_task(task_params)
            self.tasks[task_name] = task

            if self.metric_to_track is None:
                # track metric from first task in list
                self.metric_to_track = task.get_metric_to_track()
        

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

        logits = self.model_with_head(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        return loss, logits, labels


    def configure_optimizers(self):
        optimizer = OptimizerBuilder.build(self, self.optimizer_config)

        # set up scheduler
        train_len = len(self.data_module.train_dataloader())
        max_epochs = self.trainer.max_epochs
        train_steps = train_len * max_epochs

        scheduler: dict = SchedulerBuilder.build(
            optimizer,
            self.scheduler_config,
            self.metric_to_track,
            train_steps,
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
        metric_to_track, mode = self.metric_to_track
        metric_to_track = f"val_{metric_to_track}"
        filename = "{epoch}-{METRIC:.2f}".replace("METRIC", metric_to_track)

        return [
            ModelCheckpoint(
                monitor=metric_to_track,
                mode=mode,
                filename=filename,
            ),
            EarlyStopping(
                monitor=metric_to_track,
                mode=mode,
                patience=10,
            ),
        ]

