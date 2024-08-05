import os
import torch
import pytorch_lightning as pl
import pandas as pd
import random

from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Optional

PAD_TOKEN = 0
CLS_TOKEN = 1
START_TOKEN = 2
NUM_EMBEDDINGS = START_TOKEN + 15  # PAD, CLS, tokens

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

    __TOKEN_TO_IDX = {
        token: idx + START_TOKEN
        for idx, token in enumerate(__TOKENS)
    }

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
        limit_tree_height: Optional[int] = None,
        add_cls_token: bool = False,
    ):
        self.max_len = max_len
        self.augment = augment
        self.limit_tree_height = limit_tree_height
        self.add_cls_token = add_cls_token

    def __call__(self, batch):
        input_ids = []
        attention_masks = []
        labels = []

        actual_max_len = -1
        for item in batch:
            sequence, label = item["sequence"], item["label"]

            tokens = sequence.split()

            if self.limit_tree_height is not None:
                tokens = self.__limit_tree_height(tokens, self.limit_tree_height)

            if random.uniform(0, 1) < self.augment:
                tokens = self.__augment(tokens)

            # create input ids and token type ids
            if self.add_cls_token:
                indices = [CLS_TOKEN] + [self.__TOKEN_TO_IDX[token] for token in tokens]
            else:
                indices = [self.__TOKEN_TO_IDX[token] for token in tokens]

            length = min(len(indices), self.max_len)
            actual_max_len = max(actual_max_len, length)
            padding_size = self.max_len - length

            indices = indices[:length] + [PAD_TOKEN] * padding_size

            # create attention mask
            attention_mask = [1.] * length + [0.] * padding_size

            input_ids.append(indices)
            attention_masks.append(attention_mask)
            labels.append(label)

        return {
            "input_ids": torch.tensor(input_ids)[:,:actual_max_len],
            "attention_mask": torch.tensor(attention_masks)[:,:actual_max_len],
            "labels": torch.tensor(labels),
        }

    @classmethod
    def __augment(cls, tree: list[str]):
        ops: list[callable] = []
        subtokens: list[list[str]] = [[]]

        for token in tree:
            if token.startswith("["):
                ops.append(token)
                subtokens.append([])

            elif token.startswith("]"):
                random.shuffle(subtokens[-1])
                subtokens_str = " ".join(subtokens[-1])
                joined_tokens: str = f"{ops[-1]} {subtokens_str} ]"

                ops.pop()
                subtokens.pop()

                subtokens[-1].append(joined_tokens)

            else:
                subtokens[-1].append(token)
        
        return subtokens[0][0].split()
    
    @classmethod
    def __limit_tree_height(cls, tree: list[str], max_depth=5):
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

                if current_depth < max_depth:
                    subtokens[-1].extend(full_tokens)
                else:
                    subtokens[-1].extend(result_token)

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
            "augment": 1.,
            "limit_tree_height": None,
            "add_cls_token": False,
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
            "augment": 0.,
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
        self.test_dataset = ListopsDataset(data_path, "test")

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
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            **self.test_loader_config,
            shuffle=False,
        )

    def get_vocab_size(self):
        return NUM_EMBEDDINGS

    def get_pad_token_id(self):
        return PAD_TOKEN
    
    def get_cls_token_id(self):
        return CLS_TOKEN
    
