"""Download and prepare the CLINC150 dataset."""
import json
import os
from pathlib import Path

import requests


DATA_URL = (
    "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
)
DATA_DIR = Path(__file__).parent.parent / "data"


def download_clinc150():
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "data_full.json"

    if out_path.exists():
        print(f"Data already exists at {out_path}")
        return

    print("Downloading CLINC150...")
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"Saved to {out_path}")


def print_stats():
    with open(DATA_DIR / "data_full.json", encoding="utf-8") as f:
        data = json.load(f)

    for split in ["train", "val", "test"]:
        items = data[split]
        ood_items = data.get(f"oos_{split}", [])
        print(f"{split:5s}: {len(items):5d} in-domain | {len(ood_items):4d} OOD")


if __name__ == "__main__":
    download_clinc150()
    print_stats()
