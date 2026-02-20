"""CLINC150 dataset loading and preprocessing."""
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


DATA_PATH = Path(__file__).parent.parent / "data" / "data_full.json"


def load_clinc150(data_path: Path = DATA_PATH):
    """Load raw CLINC150 JSON and return split dicts."""
    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)

    splits = {}
    for split in ["train", "val", "test"]:
        in_domain = [(text, label, False) for text, label in raw[split]]
        ood = [(text, "oos", True) for text, _ in raw.get(f"oos_{split}", [])]
        splits[split] = in_domain + ood

    # Build label map from training in-domain labels only
    train_labels = sorted({label for _, label, is_ood in splits["train"] if not is_ood})
    label2id = {label: idx for idx, label in enumerate(train_labels)}

    return splits, label2id


class CLINC150Dataset(Dataset):
    """PyTorch Dataset for CLINC150."""

    def __init__(
        self,
        samples: list,
        label2id: dict,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 64,
        ood_label: int = -1,
    ):
        self.samples = samples
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ood_label = ood_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label, is_ood = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label_id = self.ood_label if is_ood else self.label2id[label]
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label_id, dtype=torch.long),
            "is_ood": torch.tensor(is_ood, dtype=torch.bool),
            "text": text,
        }
