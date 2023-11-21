import time
import numpy as np
import torch
import os

from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

class Trainer:
    """
    Receives a model with a binary classification head.
    """
    @classmethod
    def get_default_config(cls) -> dict:
        return {
            "num_workers": 4,
            "steps": None,
            "batch_size": 64,
            "learning_rate": 1e-4,
            "betas": (0.9, 0.99),
            "weight_decay": 0.,
            "eval_every": 100,
            "device": "cpu",
            "log_file": None,
            "checkpoint_file": None,
            "collator_fn": None,
        }

    def __init__(self, config, model, train_dataset, test_dataset):
        self.config = {
            **self.get_default_config(),
            **config,
        }

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def run(self):
        model, config = self.model, self.config
        model = model.to(config["device"])

        log_file = config.get("log_file")
        checkpoint_file = config.get("checkpoint_file")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

        # setup the optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            betas=config["betas"],
            weight_decay=config["weight_decay"],
        )

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            collate_fn=config["collator_fn"],
        )
        test_loader = DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            collate_fn=config["collator_fn"],
        )

        # set training mode for model
        model.train()

        # log number of params
        n_params = sum(p.numel() for p in model.parameters())
        self.__print_and_log(f"Number of parameters: {n_params}", log_file)

        # initialize training loop
        iter_num = 0
        iter_time = time.time()
        max_iters = config.get("steps")
        
        losses = []
        best_eval_loss = 1e10  # INF

        data_iter = iter(train_loader)
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            loss = self.__train_step(batch, model, optimizer)
            losses.append(loss)

            iter_num += 1

            # log and store checkpoint
            if iter_num % config["eval_every"] == 0:
                mean_training_loss = np.mean(losses)
                mean_eval_loss, mean_eval_accuracy = self.__mean_eval_loss(model, test_loader)

                tnow = time.time()
                iters_dt = tnow - iter_time
                iter_time = tnow

                self.__print_and_log(
                    f"TRAIN - {iter_num=}, {mean_training_loss=}, {mean_eval_loss=}, {mean_eval_accuracy=}, ({iters_dt:.2f}s)",
                    log_file,
                )

                if mean_eval_loss < best_eval_loss:
                    best_eval_loss = mean_eval_loss
                    torch.save(model, config["checkpoint_file"])

                losses = []
            
            # termination conditions
            if max_iters is not None and iter_num >= max_iters:
                break

    def __train_step(self, batch, model, optimizer):
        loss = self.__calc_loss_for_batch(model, batch)

        # backprop and update the parameters
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    # @torch.no_grad()
    def __mean_eval_loss(self, model, data_loader):
        losses = []
        accuracies = []
        model.eval()
        for batch in data_loader:
            loss, accuracy = self.__calc_loss_for_batch(model, batch, calc_accuracy=True)
            losses.append(loss.item())
            accuracies.append(accuracy)
        model.train()
        return np.mean(losses), np.mean(accuracies)

    def __calc_loss_for_batch(self, model, batch, calc_accuracy=False):
        input_ids = batch["input_ids"].to(self.config["device"])
        attention_mask = batch["attention_mask"].to(self.config["device"])
        labels = batch["labels"].to(self.config["device"])

        output = model(input_ids, attention_mask)
        loss = F.binary_cross_entropy(output, labels.reshape(-1, 1).float())
        if calc_accuracy is True:
            accuracy = ((output > 0.5) == labels.reshape(-1, 1)).float().mean().item()
            return loss, accuracy
        return loss


    def __print_and_log(self, message, log_file):
        print(message)
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(message + "\n")
