import pytorch_lightning as pl

from copy import deepcopy
from pytorch_lightning.callbacks import Callback
from src.data_loaders.data_module_builder import DataModuleBuilder
from src.models.model_builder import ModelBuilder
from src.tasks.task_builder import TaskBuilder
from src.training.optimizer_builder import OptimizerBuilder
from src.training.scheduler_builder import SchedulerBuilder
from src.training.callback_builder import CallbackBuilder

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

        # optimizer config
        self.optimizer_config = optimizer_params

        # scheduler config & lists
        self.scheduler_config = scheduler_params
        self.step_schedulers = []
        self.epoch_schedulers = []
        self.swa_epoch_start = None

        # callbacks
        self.callbacks_config = callbacks_params

        # build data module
        name = data_module_params.pop("_name_")
        self.data_module = DataModuleBuilder.build_data_module(
            name, data_module_params
        )

        # build model
        del model_params["_name_"]
        model_type = model_params.pop("type")
        self.model = ModelBuilder.build_model(
            model_type, model_params,
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
        

    # ======================================================== #
    #                      TRAINING                            #
    # ======================================================== #
    def training_step(self, batch, batch_idx):
        batch_size = self.__get_batch_size(self.data_module.train_dataloader())
        batch = self._preprocess_batch(batch, train=True)
        step_output = self._step(batch, batch_idx)
        self.log("train_step_loss", step_output["loss"], on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log_dict(
            {f"train_{k}": v for k, v in step_output.items()},
            on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        return step_output

    def validation_step(self, batch, batch_idx):
        batch_size = self.__get_batch_size(self.data_module.val_dataloader())
        batch = self._preprocess_batch(batch, train=False)
        step_output = self._step(batch, batch_idx)
        self.log_dict(
            {f"val_{k}": v for k, v in step_output.items()},
            on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        return step_output
    
    def test_step(self, batch, batch_idx):
        batch_size = self.__get_batch_size(self.data_module.test_dataloader())
        batch = self._preprocess_batch(batch, train=False)
        step_output = self._step(batch, batch_idx)
        self.log_dict(
            {f"test_{k}": v for k, v in step_output.items()},
            on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size
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
    
    def __get_batch_size(self, dataloader):
        return dataloader.batch_size
    
    # ======================================================== #
    #                      SCHEDULERS                          #
    # ======================================================== #
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)

        if self.swa_epoch_start is not None and self.current_epoch >= self.swa_epoch_start:
            return

        for scheduler_dict in self.step_schedulers:
            if "monitor" in scheduler_dict:
                metric_val = self.trainer.callback_metrics[scheduler_dict["monitor"]]
                scheduler_dict["scheduler"].step(metric_val)
            else:
                scheduler_dict["scheduler"].step()

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        if self.swa_epoch_start is not None and self.current_epoch >= self.swa_epoch_start:
            return

        for scheduler_dict in self.epoch_schedulers:
            if "monitor" in scheduler_dict:
                metric_val = self.trainer.callback_metrics[scheduler_dict["monitor"]]
                scheduler_dict["scheduler"].step(metric_val)
            else:
                scheduler_dict["scheduler"].step()

    # ======================================================== #
    #                      CONFIGURATION                       #
    # ======================================================== #
    def configure_optimizers(self):
        schedulers_configs = deepcopy(self.scheduler_config)
        optimizer_config = deepcopy(self.optimizer_config)

        # build optimizer
        optimizer = OptimizerBuilder.build(self, optimizer_config)

        # set up schedulers
        train_len = len(self.data_module.train_dataloader())
        max_epochs = self.trainer.max_epochs
        train_steps = train_len * max_epochs

        for scheduler_config in schedulers_configs.values():
            scheduler_dict: dict = SchedulerBuilder.build(
                optimizer,
                scheduler_config,
                train_steps,
            )

            if scheduler_dict["interval"] == "step":
                self.step_schedulers.append(scheduler_dict)
            else:
                self.epoch_schedulers.append(scheduler_dict)

        self.__configure_swa_for_schedulers()

        # return
        return optimizer
    

    def configure_callbacks(self) -> list[Callback]:
        callbacks_config = deepcopy(self.callbacks_config)
        metric_to_track, mode = self.metric_to_track
        metric_to_track = f"val_{metric_to_track}"
        return [
            CallbackBuilder.build_callback(config, metric_to_track, mode)
            for config in callbacks_config
        ]

    def __configure_swa_for_schedulers(self) -> None:
        for callback_config in self.callbacks_config:
            if callback_config["_name_"] != "stochastic_weight_averaging":
                continue

            self.swa_epoch_start = callback_config.get("swa_epoch_start", 0.8)
            if isinstance(self.swa_epoch_start, float):
                self.swa_epoch_start = int(self.swa_epoch_start * self.trainer.max_epochs)
            break


