import argparse
import sys
import json
import os
import pytorch_lightning as pl

dir_path = os.path.dirname(os.path.abspath(__file__))
project_root_path = os.path.join(dir_path, "..")
sys.path.append(project_root_path)

from src.trainers.text_classification_trainer import TextClassificationModule
from src.data_loaders.data_module_builder import DataModuleBuilder


###### Load Dataset ######
def load_data_module(data_module_params: dict):
    return DataModuleBuilder.build_data_module(**data_module_params)


###### Train ######
def run(config):
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
    module = TextClassificationModule(
        model_params=model_params,
        data_module_params=data_module_params,
        head_params=head_params,
        optimizer_params=optimizer_params,
    )

    # setuo trainer and run
    trainer = pl.Trainer(**trainer_params)

    trainer.fit(module, data_module, **fit_params)


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
        nargs="+",
        help="Path to the JSON file containing the configuration",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Load the model configuration from the provided JSON file
    configs = [parse_config(config_file_name) for config_file_name in args.config]
    
    # create required directories
    for config in configs:
        run(config)


# Run the main function
if __name__ == "__main__":
    main()
