import argparse
import sys
import json
import os
import pytorch_lightning as pl


dir_path = os.path.dirname(os.path.abspath(__file__))
project_root_path = os.path.join(dir_path, "..")
sys.path.append(project_root_path)

from src.data_loaders.data_module_builder import DataModuleBuilder
from src.trainers.trainer_builder import TrainerBuilder

###### Load Dataset ######
def load_data_module(data_module_params: dict):
    return DataModuleBuilder.build_data_module(**data_module_params)


###### Train ######
def run(config, initial_model=None):
    # unpack config
    model_params = config["model_params"]
    head_params = config["head_params"]
    data_module_params = config["data_module_params"]
    optimizer_params = config["optimizer_params"]
    trainer_params = config["trainer_params"]
    fit_params = config.get("fit_params") or {}

    # load dataset
    data_module = load_data_module(data_module_params)

    # setup trainer params
    trainer_module = TrainerBuilder.build_trainer(
        model_params=model_params,
        data_module_params=data_module_params,
        head_params=head_params,
        optimizer_params=optimizer_params,
    )

    if initial_model is not None:
        # this is a bit intrusive, but it is the easiest way to do it
        trainer_module.model.load_state_dict(initial_model.state_dict())
        trainer_module.model_with_head.model.load_state_dict(initial_model.state_dict())

    # setup trainer and run
    trainer = pl.Trainer(**trainer_params)

    # store config in log dir
    config_path = os.path.join(trainer.log_dir, "config.json")
    os.makedirs(trainer.log_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    # train
    trainer.fit(trainer_module, data_module, **fit_params)

    # returned trained model
    return trainer_module.model


###### MAIN ######

def parse_config(config_file_name):
    try:
        with open(config_file_name, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{config_file_name}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{config_file_name}' is not a valid JSON file.")
        return


def main():
    # Create the argparse parser
    parser = argparse.ArgumentParser(description="Parser for configuration")

    # Add arguments to the parser
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to the JSON file containing the configuration",
    )
    parser.add_argument(
        "-p", "--pretrain-config",
        type=str,
        help="Path to the JSON file containing the configuration for pretraining",
        required=False,
    )

    # Parse the arguments
    args = parser.parse_args()

    # Load the model configuration from the provided JSON file
    config = parse_config(args.config)
    initial_model = None

    if args.pretrain_config is not None:
        pretrain_config = parse_config(args.pretrain_config)
        initial_model = run(pretrain_config)
    
    run(config, initial_model=initial_model)


# Run the main function
if __name__ == "__main__":
    main()
