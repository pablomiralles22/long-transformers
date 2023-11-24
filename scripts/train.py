import argparse
import sys
import json
import os
import pytorch_lightning as pl

dir_path = os.path.dirname(os.path.abspath(__file__))
project_root_path = os.path.join(dir_path, "..")
sys.path.append(project_root_path)

from src.heads.classification_head import ModelWithClassificationHead
from src.models.model_builder import ModelBuilder
from src.trainers.trainer import TextClassificationModule
from src.data_loaders.data_module_builder import DataModuleBuilder


###### Load Dataset ######
def load_data_module(config):
    return DataModuleBuilder.build_data_module(**config["train_params"]["data_module"])


###### Load Model ######
def load_model(config, data_module):
    embedding_params = {
        "num_embeddings": data_module.get_vocab_size(),
        "padding_idx": data_module.get_pad_token_id(),
    }
    return ModelBuilder.build_model(config["model"], config["model_params"], embedding_params)


###### Train ######
def run(config):
    # unpack config
    train_params = config["train_params"]

    # load dataset
    data_module = load_data_module(config)

    # load model
    model = load_model(config, data_module)
    d_model = model.output_embed_dim
    model_with_class_head = ModelWithClassificationHead(model, d_model)

    # setup trainer params
    module = TextClassificationModule(model_with_class_head, train_params["optimizer_params"])

    # setuo trainer and run
    trainer = pl.Trainer(**train_params["trainer_params"])

    trainer.fit(module, data_module)


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
