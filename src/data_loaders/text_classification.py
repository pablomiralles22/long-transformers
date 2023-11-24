import os
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from typing import Literal

NUM_EMBEDDINGS = 256 + 2 # PAD, CLS, bytes
PAD_TOKEN = 0
CLS_TOKEN = 1

class TextClassificationDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "val"],
    ):
        """
        Args:
            root_dir (string): Directory with all the data.
            split (string): One of "train" or "val" to specify the split.
        """
        self.root_dir = os.path.join(root_dir, split)
        pos_dir = os.path.join(self.root_dir, "pos")
        self.pos_files = os.listdir(pos_dir)

        neg_dir = os.path.join(self.root_dir, "neg")
        self.neg_files = os.listdir(neg_dir)

    def __len__(self):
        return len(self.pos_files) + len(self.neg_files)

    def __getitem__(self, idx):
        if idx < len(self.pos_files):
            file = self.pos_files[idx]
            with open(os.path.join(self.root_dir, "pos", file), 'r', encoding='utf-8') as f:
                text = f.read()
            label = 1
        else:
            file = self.neg_files[idx - len(self.pos_files)]
            with open(os.path.join(self.root_dir, "neg", file), 'r', encoding='utf-8') as f:
                text = f.read()
            label = 0
        return {"text": text, "label": label}

class TextClassificationCollatorFn:
    def __init__(self, max_len, pad_token=0, cls_token=1):
        self.max_len = max_len
        self.pad_token = pad_token
        self.cls_token = cls_token

    def __call__(self, batch):
        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            text, label = item["text"], item["label"]
            indices = [self.cls_token] + [int(b) for b in bytes(text, encoding="utf-8")]
            length = min(len(indices), self.max_len)
            padding_size = self.max_len - length
            indices = indices[:length] + [self.pad_token] * padding_size
            attention_mask = [1.] * length + [0.] * padding_size

            input_ids.append(indices)
            attention_masks.append(attention_mask)
            labels.append(label)

        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_masks), "labels": torch.tensor(labels)}

class TextClassificationDataModule(pl.LightningDataModule):
    @classmethod
    def get_default_collator_config(cls):
        return {
            "max_len": 512,
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
            "collate_fn": TextClassificationCollatorFn(**collator_config),
        }

        # load datasets
        self.train_dataset = TextClassificationDataset(data_path, "train")
        self.val_dataset = TextClassificationDataset(data_path, "val")

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
