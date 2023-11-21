import os
import torch

from torch.utils.data import Dataset

NUM_EMBEDDINGS = 256 + 2 # PAD, CLS, bytes
PAD_TOKEN = 0
CLS_TOKEN = 1

class TextClassificationDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        """
        Args:
            root_dir (string): Directory with all the data.
            split (string): One of "train" or "test" to specify the split.
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

class CollatorFn:
    def __init__(self, max_length, pad_token=0, cls_token=1):
        self.max_length = max_length
        self.pad_token = pad_token
        self.cls_token = cls_token

    def __call__(self, batch):
        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            text, label = item["text"], item["label"]
            indices = [self.cls_token] + [int(b) for b in bytes(text, encoding="utf-8")]
            length = min(len(indices), self.max_length)
            padding_size = self.max_length - length
            indices = indices[:length] + [self.pad_token] * padding_size
            attention_mask = [1.] * length + [0.] * padding_size

            input_ids.append(indices)
            attention_masks.append(attention_mask)
            labels.append(label)

        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_masks), "labels": torch.tensor(labels)}