import os
import torch
import pytorch_lightning as pl
import pandas as pd

from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from typing import Literal

NUM_EMBEDDINGS = 2 + 17  # PAD, CLS, tokens
PAD_TOKEN = 0
CLS_TOKEN = 1

class ListopsDataset(Dataset):
    # __MAX_LEN = 2048
    __MAX_LEN = 10000

    def __init__(
        self,
        tsv_dir_path: str,
        split: Literal["train", "val", "test"],
    ):
        """
        Args:
            root_dir (string): Directory with all the data.
            split (string): One of "train" or "val" to specify the split.
        """
        self.tsv_file_path = os.path.join(tsv_dir_path, f"basic_{split}.tsv")
        self.df = pd.read_csv(self.tsv_file_path, sep='\t')
        self.df = self.df[self.df["Source"].str.split().apply(len) <= self.__MAX_LEN]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "sequence": self.df.iloc[idx]["Source"],
            "label": self.df.iloc[idx]["Target"],
        }

class ListopsCollatorFn:
    __TOKENS = [
        '(', ')',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '[MAX', '[MED', '[MIN', '[SM', ']'
    ]

    def __init__(self, max_len, pad_token=0, cls_token=1):
        self.max_len = max_len
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.__token_to_id = {token: i for i, token in enumerate(self.__TOKENS)}

    def __call__(self, batch):
        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            sequence, label = item["sequence"], item["label"]

            # create input ids
            indices = [self.cls_token] + [self.__token_to_id[token] for token in sequence.split()]

            length = min(len(indices), self.max_len)
            padding_size = self.max_len - length

            indices = indices[:length] + [self.pad_token] * padding_size

            # create attention mask
            attention_mask = [1.] * length + [0.] * padding_size

            input_ids.append(indices)
            attention_masks.append(attention_mask)
            labels.append(label)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks),
            "labels": torch.tensor(labels)
        }


class ListopsDataModule(pl.LightningDataModule):
    @classmethod
    def get_default_collator_config(cls):
        return {
            "max_len": 2048,
        }

    @classmethod
    def get_default_loader_config(cls):
        return {
            "batch_size": 16,
            "num_workers": 4,
            "pin_memory": True,
        }

    @classmethod
    def from_joint_config(cls, config: dict):
        config = deepcopy(config)
        data_path = config.pop("data_path")
        collator_config = {
            k: v for k, v in config.items()
            if k in cls.get_default_collator_config()
        }
        loader_config = {
            k: v for k, v in config.items()
            if k in cls.get_default_loader_config()
        }
        return cls(data_path, collator_config, loader_config)

    def __init__(self, data_path: str, collator_config: dict, loader_config: dict):
        super().__init__()
        self.data_path = data_path

        # build collator_fn
        collator_config = {
            **self.get_default_collator_config(),
            **collator_config,
        }

        # build loader config
        self.loader_config = {
            **self.get_default_loader_config(),
            **loader_config,
            "collate_fn": ListopsCollatorFn(**collator_config),
        }

        # load datasets
        self.train_dataset = ListopsDataset(data_path, "train")
        self.val_dataset = ListopsDataset(data_path, "val")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            **self.loader_config,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            **self.loader_config,
            shuffle=False,
        )

    def get_vocab_size(self):
        return NUM_EMBEDDINGS

    def get_pad_token_id(self):
        return PAD_TOKEN
    
    def get_cls_token_id(self):
        return CLS_TOKEN
    
