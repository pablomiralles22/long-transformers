import torch
import pytorch_lightning as pl
import torchvision

from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision import transforms


NUM_EMBEDDINGS = 256 + 2 # PAD, CLS, bytes
PAD_TOKEN = 0
CLS_TOKEN = 1

class CIFAR100CollatorFn:
    def __init__(self, max_len, pad_token=0, cls_token=1):
        self.max_len = max_len
        self.pad_token = pad_token
        self.cls_token = cls_token

    def __call__(self, batch):
        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            image, label = item
            image = (image.mean(dim=0) * 255).long().flatten().tolist()

            # build input ids
            idxs = [self.cls_token] + [2 + pixel for pixel in image]  # 2 for PAD, CLS

            length = min(len(idxs), self.max_len)
            padding_size = self.max_len - length

            idxs = idxs[:length] + [self.pad_token] * padding_size

            # build attention mask
            attention_mask = [1.] * length + [0.] * padding_size

            input_ids.append(idxs)
            attention_masks.append(attention_mask)
            labels.append(label)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.bool),
            "labels": torch.tensor(labels),
        }

class CIFAR10DataModule(pl.LightningDataModule):
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
        # torch autoaugment cifar10 policy

        # load datasets
        self.train_dataset = torchvision.datasets.CIFAR10(
            data_path,
            train=True,
            download=True,
            transform=transforms.Compose([
                # transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.01, 0.5)),
                transforms.RandomAffine(degrees=5, translate=(0.3, 0.3), scale=(0.6, 0.8), shear=20),
                transforms.ToTensor(),
                # transforms.RandomErasing(),
            ]),
        )
        self.val_dataset = torchvision.datasets.CIFAR10(
            data_path,
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            **self.loader_config,
            collate_fn=CIFAR100CollatorFn(
                **self.collator_config,
            ),
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            **self.loader_config,
            collate_fn=CIFAR100CollatorFn(
                **self.collator_config,
            ),
            shuffle=False,
        )

    def get_vocab_size(self):
        return NUM_EMBEDDINGS

    def get_pad_token_id(self):
        return PAD_TOKEN
    
    def get_cls_token_id(self):
        return CLS_TOKEN
