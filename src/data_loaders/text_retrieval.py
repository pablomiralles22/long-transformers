import os
import torch
import pytorch_lightning as pl
import random
import json

from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from typing import Literal

NUM_EMBEDDINGS = 256 + 2 # PAD, CLS, MASK, bytes
PAD_TOKEN = 0
CLS_TOKEN = 1
START_BYTE_IDX = 2

class TextRetrievalDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "val", "test"],
    ):
        """
        Args:
            root_dir (string): Directory with all the data.
            split (string): One of "train" or "val" to specify the split.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with open(os.path.join(self.root_dir, file), 'r', encoding='utf-8') as f:
            item = json.load(f)
        return item

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

class TextRetrievalCollatorFn:
    def __init__(self, max_len, fixed_start=False, augment_prob=0.):
        self.max_len = max_len
        self.fixed_start = fixed_start
        self.augment_prob = augment_prob

    def __call__(self, batch):
        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            text1, text2, label = item["text1"], item["text2"], item["label"]
            texts = (text1, text2)

            for text in texts:
                if random.random() < self.augment_prob:
                    text = augment_shuffle(text)

                start_idx = 0
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


class TextRetrievalDataModule(pl.LightningDataModule):
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
        self.train_dataset = TextRetrievalDataset(data_path, "train")
        self.val_dataset = TextRetrievalDataset(data_path, "val")
        self.test_dataset = TextRetrievalDataset(data_path, "test")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            **self.loader_config,
            collate_fn=TextRetrievalCollatorFn(
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
            collate_fn=TextRetrievalCollatorFn(
                **collator_config,
            ),
            shuffle=False,
        )
    
    def test_dataloader(self):
        collator_config = deepcopy(self.collator_config)
        collator_config["augment_prob"] = 0.
        collator_config["fixed_start"] = True

        return DataLoader(
            dataset=self.test_dataset,
            **self.loader_config,
            collate_fn=TextRetrievalCollatorFn(
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
