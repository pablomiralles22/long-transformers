import os
import torch
import pytorch_lightning as pl
import pandas as pd
import random
import math

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

    __OPERATORS = {
        "[MED": lambda x: sorted(x)[len(x) // 2],
        "[MAX": max,
        "[MIN": min,
        "[SM": lambda l : sum(l) % 10,
    }

    def __init__(
        self,
        max_len, 
        augment: float = 0.,
        pad_token_id: int = 0,
        cls_token_id: int = 1,
        pad_token_type_id: int = 0,
        cls_token_type_id: int = 1,
    ):
        self.max_len = max_len
        self.augment = augment

        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id

        self.pad_token_type_id = pad_token_type_id
        self.cls_token_type_id = cls_token_type_id

        start_id = max(pad_token_id, cls_token_id) + 1
        self.__token_to_id = {
            token: i + start_id
            for i, token in enumerate(self.__TOKENS)
        }

    def __call__(self, batch):
        input_ids = []
        attention_masks = []
        # token_type_ids = []
        labels = []

        for item in batch:
            sequence, label = item["sequence"], item["label"]

            tokens = sequence.split()
            if random.uniform(0, 1) < self.augment:
                tokens = self.__augment(tokens)

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

    @classmethod
    def __augment(cls, tree: list[str], a=-8., b=0.5):
        ops: list[callable] = []
        operands: list[list[int]] = [[]]
        subtokens: list[list[str]] = [[]]
        current_depth = 1

        for token in tree:
            if token.startswith("["):
                ops.append(token)
                operands.append([])
                subtokens.append([])
                current_depth += 1

            elif token.startswith("]"):
                result = cls.__OPERATORS[ops[-1]](operands[-1])
                
                full_tokens: list[str] = [ops[-1]] + subtokens[-1] + ["]"]
                result_token: list[str] = [str(result)]

                ops.pop()
                operands.pop()
                subtokens.pop()

                logit_contract = a + b * current_depth
                prob_contract = 1. / (1. + math.exp(-logit_contract))
                if random.uniform(0, 1) < prob_contract:
                    subtokens[-1].extend(result_token)
                else:
                    subtokens[-1].extend(full_tokens)

                operands[-1].append(result)

                current_depth -= 1
            else:
                operands[-1].append(int(token))
                subtokens[-1].append(token)
        
        return subtokens[0]


class ListopsDataModule(pl.LightningDataModule):
    @classmethod
    def get_default_collator_config(cls):
        return {
            "max_len": 2000,
            "augment": False,
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
        train_collator_config = {
            **self.get_default_collator_config(),
            **collator_config,
        }
        test_collator_config = {
            **train_collator_config,
            "augment": False,
        }

        # build loader config
        self.train_loader_config = {
            **self.get_default_loader_config(),
            **loader_config,
            "collate_fn": ListopsCollatorFn(**train_collator_config),
        }
        self.test_loader_config = {
            **self.get_default_loader_config(),
            **loader_config,
            "collate_fn": ListopsCollatorFn(**test_collator_config),
        }

        # load datasets
        self.train_dataset = ListopsDataset(data_path, "train")
        self.val_dataset = ListopsDataset(data_path, "val")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            **self.train_loader_config,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            **self.test_loader_config,
            shuffle=False,
        )

    def get_vocab_size(self):
        return NUM_EMBEDDINGS

    def get_pad_token_id(self):
        return self.train_loader_config["collate_fn"].pad_token_id
    
    def get_cls_token_id(self):
        return self.train_loader_config["collate_fn"].cls_token_id
    
