import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback
from src.data_loaders.data_module_builder import DataModuleBuilder
from src.models.model_builder import ModelBuilder
from src.tasks.task_builder import TaskBuilder
from src.utils.optimizer_builder import OptimizerBuilder
from src.utils.scheduler_builder import SchedulerBuilder
from src.utils.callback_builder import CallbackBuilder

class TrainerModule(pl.LightningModule):
    def __init__(
        self,
        model_params: dict,
        data_module_params: dict,
        tasks_params: list[dict],
        optimizer_params: dict,
        scheduler_params: dict,
        callbacks_params: list[dict],
    ):
        super().__init__()
        self.save_hyperparameters()

        # optimizer config
        self.optimizer_config = optimizer_params

        # scheduler config
        self.scheduler_config = scheduler_params

        # callbacks
        self.callbacks_config = callbacks_params

        # build data module
        name = data_module_params.pop("_name_")
        self.data_module = DataModuleBuilder.build_data_module(
            name, data_module_params
        )

        # build model
        del model_params["_name_"]
        type = model_params.pop("type")
        self.model = ModelBuilder.build_model(
            type, model_params,
            vocab_size=self.data_module.get_vocab_size(),
            padding_idx=self.data_module.get_pad_token_id(),
        )

        # build tasks
        self.metric_to_track = None
        self.tasks = dict()

        for task_params in tasks_params:
            task_name = task_params["_name_"]
            task_params["head_params"]["input_dim"] = self.model.get_output_embedding_dim()

            task = TaskBuilder.build_task(task_params)
            self.tasks[task_name] = task
            # need to set a property for each task so it is moved to the right device and type
            setattr(self, f"task_{task_name}", task)

            if self.metric_to_track is None:
                # track metric from first task in list
                self.metric_to_track = task.get_metric_to_track()
        

    def training_step(self, batch, batch_idx):
        batch = self._preprocess_batch(batch, train=True)
        step_output = self._step(batch, batch_idx)
        self.log("train_step_loss", step_output["loss"], on_step=True, on_epoch=False, prog_bar=True)
        self.log_dict(
            {f"train_{k}": v for k, v in step_output.items()},
            on_step=False, on_epoch=True, prog_bar=True
        )
        return step_output

    def validation_step(self, batch, batch_idx):
        batch = self._preprocess_batch(batch, train=False)
        step_output = self._step(batch, batch_idx)
        self.log_dict(
            {f"val_{k}": v for k, v in step_output.items()},
            on_step=False, on_epoch=True, prog_bar=True
        )
        return step_output

    def _preprocess_batch(self, batch, train: bool):
        for _, task in self.tasks.items():
            batch = task.preprocess_input(batch, train=train)
        return batch

    def _step(self, batch, _):
        outputs = self.model(batch["input_ids"], batch["attention_mask"])
        step_output = dict()
        for task_name, task in self.tasks.items():
            task_outputs = task(batch, outputs)
            for key, value in task_outputs.items():
                step_output[f"{task_name}_{key}"] = value
        step_output["loss"] = sum([v for k, v in step_output.items() if "loss" in k])
        return step_output


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
            "lr_scheduler": scheduler,
        }
    

    def configure_callbacks(self) -> list[Callback]:
        metric_to_track, mode = self.metric_to_track
        metric_to_track = f"val_{metric_to_track}"
        return [
            CallbackBuilder.build_callback(config, metric_to_track, mode)
            for config in self.callbacks_config
        ]

