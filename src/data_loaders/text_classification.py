import os
import torch
import pytorch_lightning as pl
import random

from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from typing import Literal

NUM_EMBEDDINGS = 256 + 2 # PAD, CLS, MASK, bytes
PAD_TOKEN = 0
CLS_TOKEN = 1
START_BYTE_IDX = 2

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

def augment_shuffle(text: str):
    words = text.split()
    random.shuffle(words)
    return " ".join(words)

def augment_random_changes(text: str):
    idxs = random.sample(range(len(text)), int(0.2 * len(text)))
    text = list(text)
    for idx in idxs:
        text[idx] = chr(random.randint(0, 255))
    return "".join(text)

class TextClassificationCollatorFn:
    def __init__(self, max_len, fixed_start=False, augment_prob=0.):
        self.max_len = max_len
        self.fixed_start = fixed_start
        self.augment_prob = augment_prob

    def __call__(self, batch):
        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            text, label = item["text"], item["label"]

            if random.random() < self.augment_prob:
                # augment_fn = random.choice([augment_shuffle, augment_random_changes])
                text = augment_shuffle(text)

            start_idx = 0
            # if self.fixed_start is False and len(text) > self.max_len:
            #     start_idx = torch.randint(0, len(text) - self.max_len + 1, (1,)).item()
            if self.fixed_start is False:
                start_idx = torch.randint(0, max(1, len(text) - 64), (1,)).item()

            text = text[start_idx:start_idx + self.max_len]

            # build input ids
            text_idxs = [START_BYTE_IDX + int(b) for b in bytes(text, encoding="utf-8")]  # 3 for PAD, CLS, MASK

            indices = [CLS_TOKEN] + text_idxs
            length = min(len(indices), self.max_len)
            padding_size = self.max_len - length
            indices = indices[:length] + [PAD_TOKEN] * padding_size

            # build attention mask
            attention_mask = [1.] * length + [0.] * padding_size

            input_ids.append(indices)
            attention_masks.append(attention_mask)
            labels.append(label)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.bool),
            "labels": torch.tensor(labels),
        }

class TextClassificationDataModule(pl.LightningDataModule):
    @classmethod
    def get_default_collator_config(cls):
        return {
            "max_len": 512,
            "fixed_start": False,
            "augment_prob": 0.,
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
        self.collator_config = {
            **self.get_default_collator_config(),
            **collator_config,
        }

        # build loader config
        self.loader_config = {
            **self.get_default_loader_config(),
            **loader_config,
        }

        # load datasets
        self.train_dataset = TextClassificationDataset(data_path, "train")
        self.val_dataset = TextClassificationDataset(data_path, "val")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            **self.loader_config,
            collate_fn=TextClassificationCollatorFn(
                **self.collator_config,
            ),
            shuffle=True,
        )

    def val_dataloader(self):
        collator_config = deepcopy(self.collator_config)
        collator_config["augment_prob"] = 0.
        collator_config["fixed_start"] = True

        return DataLoader(
            dataset=self.val_dataset,
            **self.loader_config,
            collate_fn=TextClassificationCollatorFn(
                **collator_config,
            ),
            shuffle=False,
        )

    def get_vocab_size(self):
        return NUM_EMBEDDINGS

    def get_pad_token_id(self):
        return PAD_TOKEN
    
    def get_cls_token_id(self):
        return CLS_TOKEN
