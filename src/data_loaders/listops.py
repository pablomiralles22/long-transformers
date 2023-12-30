import os
import torch
import pytorch_lightning as pl
import pandas as pd

from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from typing import Literal

NUM_EMBEDDINGS = 2 + 15  # PAD, CLS, tokens

class ListopsDataset(Dataset):
    def __init__(
        self,
        tsv_dir_path: str,
        split: Literal["train", "val", "test"],
        compressed: bool = True,
    ):
        """
        Args:
            root_dir (string): Directory with all the data.
            split (string): One of "train" or "val" to specify the split.
        """
        file_name = f"basic_{split}.tsv" if compressed is False else f"basic_{split}_compressed.tsv"
        self.tsv_file_path = os.path.join(tsv_dir_path, file_name)
        self.df = pd.read_csv(self.tsv_file_path, sep='\t')

        if compressed is False:
            self.df["Source"] = self.df["Source"].str.replace(r'[()]', '', regex=True)
            self.df["Source"] = self.df["Source"].str.replace(r'\s+', ' ', regex=True)
            self.df["Source"] = self.df["Source"].str.strip()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "sequence": self.df.iloc[idx]["Source"],
            "label": self.df.iloc[idx]["Target"],
        }

class ListopsCollatorFn:
    __TOKENS = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '[MAX', '[MED', '[MIN', '[SM', ']',
    ]

    def __init__(
        self,
        max_len, 
        pad_token_id: int = 0,
        cls_token_id: int = 1,
        pad_token_type_id: int = 0,
        cls_token_type_id: int = 1,
    ):
        self.max_len = max_len

        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id

        self.pad_token_type_id = pad_token_type_id
        self.cls_token_type_id = cls_token_type_id

        start_id = max(pad_token_id, cls_token_id) + 1
        self.__token_to_id = {
            token: i
            for i, token in enumerate(self.__TOKENS, start=start_id)
        }

    def __call__(self, batch):
        input_ids = []
        attention_masks = []
        token_type_ids = []
        labels = []

        for item in batch:
            sequence, label = item["sequence"], item["label"]

            tokens = sequence.split()

            # create input ids and token type ids
            indices = [self.cls_token_id] + [self.__token_to_id[token] for token in tokens]
            # token_type_ids = self.__get_token_type_ids(tokens)

            length = min(len(indices), self.max_len)
            padding_size = self.max_len - length

            indices = indices[:length] + [self.pad_token_id] * padding_size
            # token_type_ids = token_type_ids[:length] + [self.pad_token_type_id] * padding_size

            # create attention mask
            attention_mask = [1.] * length + [0.] * padding_size

            input_ids.append(indices)
            attention_masks.append(attention_mask)
            labels.append(label)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks),
            # "token_type_ids": torch.tensor(token_type_ids),
            "labels": torch.tensor(labels)
        }
    
    def __get_token_type_ids(self, tokens: list[str]):
        token_type_ids = [self.cls_token_type_id]
        token_queue = []
        for token in tokens:
            if token.startswith("[") or token.startswith("("):
                token_queue.append(token)
            elif token.endswith("]") or token.endswith(")"):
                token_queue.pop()
            token_type_ids.append(len(token_queue) + self.cls_token_type_id)
        return token_type_ids


class ListopsDataModule(pl.LightningDataModule):
    @classmethod
    def get_default_collator_config(cls):
        return {
            "max_len": 2000,
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
        return self.loader_config["collate_fn"].pad_token_id
    
    def get_cls_token_id(self):
        return self.loader_config["collate_fn"].cls_token_id
    
