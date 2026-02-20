"""Evaluate all OOD detection methods on CLINC150 test set."""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from dataset import load_clinc150, CLINC150Dataset
from models import IntentClassifier, MCDropoutClassifier
from metrics import compute_all_metrics
from methods.msp import compute_msp_scores
from methods.energy import compute_energy_scores
from methods.mahalanobis import fit_mahalanobis, compute_mahalanobis_scores
from methods.knn import fit_knn, compute_knn_scores
from methods.mc_dropout import compute_mc_dropout_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="checkpoints/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--knn_k", type=int, default=1)
    parser.add_argument("--mc_passes", type=int, default=20)
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt["model_name"]
    num_classes = ckpt["num_classes"]
    label2id = ckpt["label2id"]

    model = MCDropoutClassifier(model_name, num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer, label2id


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer, label2id = load_model(args.model_path, device)
    num_classes = len(label2id)

    splits, _ = load_clinc150()

    train_ds = CLINC150Dataset(splits["train"], label2id, tokenizer, args.max_length)
    test_ds = CLINC150Dataset(splits["test"], label2id, tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Ground truth: is_ood for test set
    is_ood_gt = np.array([int(s[2]) for s in splits["test"]])

    results = {}

    # --- MSP ---
    print("Computing MSP scores...")
    msp_scores = compute_msp_scores(model, test_loader, device).numpy()
    results["MSP"] = compute_all_metrics(msp_scores, is_ood_gt)

    # --- Energy ---
    print("Computing Energy scores...")
    energy_scores = compute_energy_scores(model, test_loader, device).numpy()
    results["Energy"] = compute_all_metrics(energy_scores, is_ood_gt)

    # --- Mahalanobis ---
    print("Fitting Mahalanobis...")
    class_means, precision = fit_mahalanobis(model, train_loader, num_classes, device)
    maha_scores = compute_mahalanobis_scores(
        model, test_loader, class_means, precision, device
    ).numpy()
    results["Mahalanobis"] = compute_all_metrics(maha_scores, is_ood_gt)

    # --- KNN ---
    print("Fitting KNN...")
    train_feats = fit_knn(model, train_loader, device)
    knn_scores = compute_knn_scores(
        model, test_loader, train_feats, device, k=args.knn_k
    ).numpy()
    results[f"KNN (k={args.knn_k})"] = compute_all_metrics(knn_scores, is_ood_gt)

    # --- MC Dropout ---
    print("Computing MC Dropout scores...")
    mc_scores = compute_mc_dropout_scores(
        model, test_loader, device, n_passes=args.mc_passes
    ).numpy()
    results[f"MC Dropout (n={args.mc_passes})"] = compute_all_metrics(mc_scores, is_ood_gt)

    # --- Print results table ---
    print("\n" + "=" * 65)
    print(f"{'Method':<25} {'AUROC':>8} {'FPR@95':>8} {'AUPR':>8}")
    print("=" * 65)

    # Published SotA from Podolskiy et al. (2021) for reference
    sota = {"Mahalanobis (Podolskiy 2021)": (0.9676, 0.1832, None)}
    for method, metrics_ref in sota.items():
        auroc_v, fpr_v, aupr_v = metrics_ref
        aupr_str = f"{aupr_v:.4f}" if aupr_v else "  N/A  "
        print(f"  {method:<23} {auroc_v:>8.4f} {fpr_v:>8.4f} {aupr_str:>8}  [published]")
    print("-" * 65)

    for method, m in results.items():
        print(
            f"  {method:<23} {m['AUROC']:>8.4f} {m['FPR@95TPR']:>8.4f} {m['AUPR']:>8.4f}"
        )
    print("=" * 65)

    # Save results
    import json
    out_path = Path("results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
