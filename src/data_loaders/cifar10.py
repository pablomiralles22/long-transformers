import torch
import torch.nn as nn
import random
import pytorch_lightning as pl
import torchvision

from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision import transforms
# from torchvision.transforms import v2

# class RandomCifar10Augmentator(nn.Module):
#     __AUGMENTATIONS = [
#         # transforms.RandomPosterize(bits=4, p=1.0),
#         transforms.RandomSolarize(threshold=128.0, p=1.0),
#         transforms.RandomEqualize(p=1.0),
#         transforms.RandomInvert(p=1.0),
#         # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
#         # transforms.RandomAdjustSharpness(sharpness_factor=2, p=1.0),
#         # transforms.RandomAutocontrast(p=1.0),
#         transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
#         transforms.RandomAutocontrast(p=0.0),  # identity
#     ]

#     # Function to apply a random augmentation
#     def forward(self, img):
#         augmentation = random.choice(self.__AUGMENTATIONS)
#         return augmentation(img)


NUM_EMBEDDINGS = 256 + 2 # PAD, CLS, bytes
PAD_TOKEN = 0
CLS_TOKEN = 1
START_TOKEN = 2

class CIFAR100CollatorFn:
    def __init__(self, max_len, augment=True, add_cls_token=False):
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Grayscale(num_output_channels=1),
            transforms.PILToTensor(),
        ]) if augment is True else transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.PILToTensor(),
        ])
        self.add_cls_token = add_cls_token

    def __call__(self, batch):
        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            image, label = item
            image = self.transform(image)
            image = torch.flatten(image).long()  # Important to prevent overflow

            # build list of indexes
            pixels = (START_TOKEN + image).tolist()
            assert all([2 <= pixel <= 257 for pixel in pixels]), f"pixels={pixels}"

            if self.add_cls_token:
                idxs = [CLS_TOKEN] + pixels
            else:
                idxs = pixels
            # assert len(idxs) == 32 * 32 + 1, f"len(idxs)={len(idxs)}"

            length = min(len(idxs), self.max_len)
            padding_size = self.max_len - length

            idxs = idxs[:length] + [PAD_TOKEN] * padding_size

            # build attention mask
            attention_mask = [True] * length + [False] * padding_size

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
            "add_cls_token": False,
            "augment": True,
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
        self.train_val_dataset = torchvision.datasets.CIFAR10(
            data_path,
            train=True,
            download=True,
        )

        train_len = int(0.9 * len(self.train_val_dataset))
        val_len = len(self.train_val_dataset) - train_len

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_val_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42),
        )

        self.test_dataset = torchvision.datasets.CIFAR10(
            data_path,
            train=False,
            download=True,
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
        collator_config = deepcopy(self.collator_config)
        collator_config["augment"] = False
        return DataLoader(
            dataset=self.val_dataset,
            **self.loader_config,
            collate_fn=CIFAR100CollatorFn(
                **collator_config,
            ),
            shuffle=False,
        )

    def test_dataloader(self):
        collator_config = deepcopy(self.collator_config)
        collator_config["augment"] = False
        return DataLoader(
            dataset=self.test_dataset,
            **self.loader_config,
            collate_fn=CIFAR100CollatorFn(
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
