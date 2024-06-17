import os
import torch
import pytorch_lightning as pl
import random
import json

from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from typing import Literal

NUM_EMBEDDINGS = 256 + 3 # PAD, CLS, MASK, bytes
PAD_TOKEN = 0
CLS_TOKEN = 1
MASK_TOKEN = 2
START_BYTE_IDX = 3

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
        if split == "val": split = "eval"
        self.root_dir = os.path.join(root_dir, split)
        self.files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with open(os.path.join(self.root_dir, file), 'r', encoding='utf-8') as f:
            item = json.load(f)
        return item


class TextRetrievalCollatorFn:
    def __init__(self, max_len, random_start=True, mask_ratio: float=0.3, random_ratio: float=0.33):
        self.max_len = max_len
        self.random_start = random_start
        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio


    def __call__(self, batch):
        input_ids = []
        corrupted_input_ids = []
        attention_masks = []
        labels = []

        batch_lengths = [
            len(bytes(item[label], encoding="utf-8"))
            for item in batch for label in ["text1", "text2"]
        ]
        batch_max_len = min(self.max_len, max(batch_lengths))
        for item in batch:
            text1, text2, label = item["text1"], item["text2"], item["label"]
            texts = (text1, text2)

            for text in texts:
                start_idx = (
                    random.randint(0, max(1, len(text) - batch_max_len))
                    if self.random_start is True
                    else 0
                )

                # indices
                text = text[start_idx:start_idx + batch_max_len]
                text_idxs = [START_BYTE_IDX + int(b) for b in bytes(text, encoding="utf-8")]  # 3 for PAD, CLS, MASK
                indices = [CLS_TOKEN] + text_idxs

                # corrupt data
                corrupt_indices = indices.copy()
                mask_idxs = random.sample(range(1, len(corrupt_indices)), int(self.mask_ratio * len(corrupt_indices)))
                for idx in mask_idxs:
                    if random.random() < self.random_ratio:
                        corrupt_indices[idx] = random.randint(START_BYTE_IDX, NUM_EMBEDDINGS - 1)
                    else:
                        corrupt_indices[idx] = MASK_TOKEN

                # cut and pad tokens
                length = min(len(indices), batch_max_len)
                padding_size = batch_max_len - length
                indices = indices[:length] + [PAD_TOKEN] * padding_size
                corrupt_indices = corrupt_indices[:length] + [PAD_TOKEN] * padding_size

                # build attention mask
                attention_mask = [1.] * length + [0.] * padding_size

                input_ids.append(indices)
                corrupted_input_ids.append(corrupt_indices)
                attention_masks.append(attention_mask)

            labels.append(label)

        return {
            "input_ids": torch.tensor(input_ids),
            "corrupted_input_ids": torch.tensor(corrupted_input_ids),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.bool),
            "labels": torch.tensor(labels),
        }


class TextRetrievalDataModule(pl.LightningDataModule):
    @classmethod
    def get_default_collator_config(cls):
        return {
            "max_len": 512,
            "random_start": True,
            "mask_ratio": 0.3,
            "random_ratio": 0.33,
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
        collator_config["mask_ratio"] = 0.
        collator_config["random_start"] = False

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
        collator_config["mask_ratio"] = 0.
        collator_config["random_start"] = False

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
