import argparse
import torch
import sys
import json
import os
import numpy as np

from torch.utils.data import DataLoader
from torch.nn import functional as F

sys.path.append(
    os.path.join(os.getcwd(), "src")
)  # WARNING: this migh file depending on your location

from utils import ModelWithClassificationHead
from logger import print_and_log
from models.conv_transformer import ConvTransformer 
from models.transformer_base import TransformerModel
from task_data.text_classification import (
    TextClassificationDataset,
    Collator,
    NUM_EMBEDDINGS as TEXT_CLASSIFICATION_NUM_EMBEDDINGS,
    PAD_TOKEN as TEXT_CLASSIFICATION_PAD_TOKEN,
)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

###### Load Model ######
def load_conv_transformer(model_params, num_embeddings, pad_token):
    embedding_params = {
        "num_embeddings": num_embeddings,
        "embedding_dim": model_params["conv_layers_params"][0]["conv_params"][
            "in_channels"
        ],
        "padding_idx": pad_token,
    }
    return ConvTransformer(
        embedding_params,
        model_params["conv_layers_params"],
        model_params["transformer_params"],
    ), model_params["transformer_params"]["layer_params"]["d_model"]

def load_transformer(model_params, num_embeddings, pad_token):
    embedding_params = {
        "num_embeddings": num_embeddings,
        "embedding_dim": model_params["layer_params"]["d_model"],
        "padding_idx": pad_token,
    }
    return TransformerModel(
        embedding_params,
        model_params["layer_params"],
        model_params["num_layers"],
    ), model_params["layer_params"]["d_model"]


def load_model(config):
    model_file_name = config["model_file_name"]
    if os.path.exists(model_file_name):
        model = torch.load(model_file_name)
        print("Model loaded from disk.")
        return model

    task = config["task"]
    match task:
        case "text-classification":
            num_embeddings = TEXT_CLASSIFICATION_NUM_EMBEDDINGS
            pad_token = TEXT_CLASSIFICATION_PAD_TOKEN
        case _:
            raise ValueError(f"Unknown task: {task}")

    match config["model"]:
        case "conv-transformer":
            return load_conv_transformer(config["model_params"], num_embeddings, pad_token)
        case "transformer":
            return load_transformer(config["model_params"], num_embeddings, pad_token)
        case _:
            raise ValueError(f"Unknown model: {config['model']}")


###### Load Dataset ######
def load_text_classification_dataset(config):
    train_set = TextClassificationDataset(config["dataset_root_dir"], split="train")
    test_set = TextClassificationDataset(config["dataset_root_dir"], split="test")

    train_params = config["train_params"]

    train_params = {
        "batch_size": train_params["batch_size"],
        "shuffle": True,
        "num_workers": 0,
        "collate_fn": Collator(train_params["max_len"]),
    }

    test_params = {
        **train_params,
        "shuffle": False,
    }

    training_loader = DataLoader(train_set, **train_params)
    testing_loader = DataLoader(test_set, **test_params)
    return training_loader, testing_loader


def load_dataset(config):
    match config["task"]:
        case "text-classification":
            return load_text_classification_dataset(config)
        case _:
            raise ValueError(f"Unknown task: {config['task']}")


###### Train ######


def train(model, epoch, training_loader, optimizer, log_file_name):
    model.train()
    for ind, item in enumerate(training_loader):
        input_ids, attention_mask, labels = (
            item["input_ids"].to(device),
            item["attention_mask"].to(device),
            item["labels"].to(device),
        )

        loss = F.binary_cross_entropy(
            model(input_ids, attention_mask), labels.reshape(-1, 1).float()
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if ind % 500 == 0:
            print_and_log(log_file_name, f"TRAIN - {epoch=}, {ind=}, loss={loss.item()}")

def test(model, epoch, testing_loader, log_file_name):
    losses = []
    accuracies = []
    with torch.no_grad():
        for _, item in enumerate(testing_loader):
            input_ids, attention_mask, labels = (
                item["input_ids"].to(device),
                item["attention_mask"].to(device),
                item["labels"].to(device),
            )
            output = model(input_ids, attention_mask)

            loss = F.binary_cross_entropy(output, labels.reshape(-1, 1).float())
            losses.append(loss.item())
            accurcy = ((output > 0.5) == labels.reshape(-1, 1)).float().mean()
            accuracies.append(accurcy.item())

    mean_loss = np.mean(losses)
    mean_accuracy = np.mean(accuracies)

    print_and_log(log_file_name, f"EVAL - {epoch=}, {mean_loss=}, {mean_accuracy=}")
    return np.mean(losses)


def run(config):
    # load model
    model, d_model = load_model(config)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    model = ModelWithClassificationHead(model, d_model).to(device)

    # load dataset
    training_loader, testing_loader = load_dataset(config)

    # create optimizer
    train_params = config["train_params"]
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=train_params["lr"],
        weight_decay=train_params["weight_decay"],
    )

    # train
    best_test_loss = 1e10
    for epoch in range(train_params["epochs"]):
        train(model, epoch, training_loader, optimizer, config["log_file_name"])
        if epoch % 5 == 0:
            test_loss = test(model, epoch, testing_loader, config["log_file_name"])
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.model, config["model_file_name"])


###### MAIN ######


def main():
    # Create the argparse parser
    parser = argparse.ArgumentParser(description="Parser for configuration")

    # Add arguments to the parser
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON file containing the configuration",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Load the model configuration from the provided JSON file
    try:
        with open(args.config, "r", encoding="utf-8") as file:
            config = json.load(file)
            print("Model configuration loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{args.config}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{args.config}' is not a valid JSON file.")
        return
    
    # create required directories
    os.makedirs(os.path.dirname(config["model_file_name"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["log_file_name"]), exist_ok=True)

    run(config)


# Run the main function
if __name__ == "__main__":
    main()
