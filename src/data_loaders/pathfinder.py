import torch
import pytorch_lightning as pl
import os
import glob
import pandas as pd

from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from copy import deepcopy
from torchvision import transforms


NUM_EMBEDDINGS = 256 + 2  # PAD, CLS, bytes
PAD_TOKEN = 0
CLS_TOKEN = 1


class PathfinderSubdataset(Dataset):
    def __init__(self, data_path, dir_id):
        self.img_dir = os.path.join(data_path, "imgs", str(dir_id))

        # load the metadata
        metadata = pd.read_csv(
            os.path.join(data_path, "metadata", f"{dir_id}.npy"), sep=" ", header=None
        )
        self.labels = metadata.iloc[:, 3].values
        self.num_imgs = len(self.labels)

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        label = self.labels[idx]
        try:
            img = Image.open(os.path.join(self.img_dir, f"sample_{idx}.png"))
        except FileNotFoundError:
            img = None
        return img, label


def build_dataset(data_path):
    if not isinstance(data_path, list):
        data_path = [data_path]

    datasets = []
    for path in data_path:
        num_dirs = len(glob.glob(os.path.join(path, "metadata", "*.npy")))
        datasets.extend(PathfinderSubdataset(path, dir_id) for dir_id in range(num_dirs))

    return ConcatDataset(datasets)


class PathfinderCollatorFn:
    def __init__(self, max_len, enable_augment=False, pad_token=0, cls_token=1):
        self.max_len = max_len
        self.pad_token = pad_token
        self.cls_token = cls_token
        if enable_augment is True:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 0.9)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.PILToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.PILToTensor(),
                ]
            )

    def __call__(self, batch):
        input_ids = []
        attention_masks = []
        labels = []

        for image, label in batch:
            if image is None:
                continue
            image = self.transform(image)
            image = torch.flatten(image).long()  # Important to prevent overflow

            # build input ids
            idxs = [self.cls_token] + [2 + pixel for pixel in image]  # 2 for PAD, CLS

            length = min(len(idxs), self.max_len)
            padding_size = self.max_len - length

            idxs = idxs[:length] + [self.pad_token] * padding_size

            # build attention mask
            attention_mask = [1.0] * length + [0.0] * padding_size

            input_ids.append(idxs)
            attention_masks.append(attention_mask)
            labels.append(label)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.bool),
            "labels": torch.tensor(labels),
        }


class PathfinderDataModule(pl.LightningDataModule):
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
        config = deepcopy(config)
        data_path = config.pop("data_path")
        collator_config = {
            k: v for k, v in config.items() if k in cls.get_default_collator_config()
        }
        loader_config = {
            k: v for k, v in config.items() if k in cls.get_default_loader_config()
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
        dataset = build_dataset(data_path)
        # 0.8 train, 0.2 val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            **self.loader_config,
            collate_fn=PathfinderCollatorFn(
                **self.collator_config,
                enable_augment=True,
            ),
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            **self.loader_config,
            collate_fn=PathfinderCollatorFn(
                **self.collator_config,
                enable_augment=False,
            ),
            shuffle=False,
        )

    def get_vocab_size(self):
        return NUM_EMBEDDINGS

    def get_pad_token_id(self):
        return PAD_TOKEN

    def get_cls_token_id(self):
        return CLS_TOKEN
