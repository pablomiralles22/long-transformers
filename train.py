import hydra
import sys
import json
import os
import pytorch_lightning as pl
import wandb


dir_path = os.path.dirname(os.path.abspath(__file__))
project_root_path = os.path.join(dir_path, "..")
sys.path.append(project_root_path)

from src.trainers.general_trainer import TrainerModule
from omegaconf import OmegaConf


###### Train ######
def run(config):
    # unpack config
    model_params = config["model"]
    tasks_params = list(config["task"].values())
    data_module_params = config["dataset"]
    optimizer_params = config["optimizer"]
    scheduler_params = config["scheduler"]
    trainer_params = config["trainer"]
    callback_params = list(config.get("callback").values()) or []
    fit_params = config.get("fit_params") or {}
    wandb_params = config.get("wandb") or {}


    # setup trainer params
    trainer_module = TrainerModule(
        model_params=model_params,
        data_module_params=data_module_params,
        tasks_params=tasks_params,
        optimizer_params=optimizer_params,
        scheduler_params=scheduler_params,
        callbacks_params=callback_params,
    )

    # get dataset
    data_module = trainer_module.data_module

    # setup trainer and run
    if wandb_params is not None:
        logger = pl.loggers.WandbLogger(
            config=config,
            settings=wandb.Settings(start_method="fork"),
            **wandb_params,
        )
    else:
        logger = None

    trainer = pl.Trainer(**trainer_params, logger=logger)

    # store config in log dir
    if logger is None:
        config_path = os.path.join(trainer.log_dir, "config.json")
        os.makedirs(trainer.log_dir, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)

    # train
    trainer.fit(trainer_module, data_module, **fit_params)

@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):
    config = OmegaConf.to_object(config)
    # print(OmegaConf.to_yaml(config))
    run(config)

if __name__ == "__main__":
    main()
