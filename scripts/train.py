import argparse
import torch
import sys
import json
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
project_root_path = os.path.join(dir_path, "..")
sys.path.append(project_root_path)

from src.heads.classification_head import ModelWithClassificationHead
from src.models.model_builder import ModelBuilder
from src.trainers.trainer import Trainer
from src.data_loaders.text_classification import (
    TextClassificationDataset,
    CollatorFn as TextClassificationCollatorFn,
    NUM_EMBEDDINGS as TEXT_CLASSIFICATION_NUM_EMBEDDINGS,
    PAD_TOKEN as TEXT_CLASSIFICATION_PAD_TOKEN,
)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

###### Load Model ######
def load_model(config):
    model_file_name = config["train_params"]["checkpoint_file"]
    if os.path.exists(model_file_name):
        model = torch.load(model_file_name)
        print("Model loaded from disk.")
        return model

    task = config["task_params"]["task"]
    match task:
        case "text-classification":
            embedding_params = {
                "num_embeddings": TEXT_CLASSIFICATION_NUM_EMBEDDINGS,
                "padding_idx": TEXT_CLASSIFICATION_PAD_TOKEN,
            }
        case _:
            raise ValueError(f"Unknown task: {task}")

    return ModelBuilder.build_model(config["model"], config["model_params"], embedding_params)


###### Load Dataset ######
def load_text_classification_dataset(config):
    train_set = TextClassificationDataset(config["task_params"]["dataset_root_dir"], split="train")
    test_set = TextClassificationDataset(config["task_params"]["dataset_root_dir"], split="test")
    collator_fn = TextClassificationCollatorFn(config["train_params"]["max_len"])

    return train_set, test_set, collator_fn


def load_dataset(config):
    match config["task_params"]["task"]:
        case "text-classification":
            return load_text_classification_dataset(config)
        case _:
            raise ValueError(f"Unknown task: {config['task']}")


###### Train ######
def run(config):
    # load model
    model = load_model(config)
    d_model = model.output_embed_dim
    model = ModelWithClassificationHead(model, d_model).to(device)

    # load dataset
    train_set, test_set, collator_fn = load_dataset(config)

    # setup trainer params
    train_params = config["train_params"]
    train_params["collator_fn"] = collator_fn
    if torch.cuda.is_available() is False:
        train_params["device"] =  "cpu"

    # setuo trainer and run
    trainer = Trainer(train_params, model, train_set, test_set)
    trainer.run()


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
