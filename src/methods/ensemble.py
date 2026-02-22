"""MahaKNN: Calibrated Ensemble of Mahalanobis and KNN OOD scores.

Our method. Combines Mahalanobis Distance and k-NN similarity scores
via a convex combination with weight alpha selected on the validation set.

score(x) = alpha * maha_score(x) + (1 - alpha) * knn_score(x)

Both scores are first standardised (zero mean, unit std) on the val set
so they are on the same scale before combining.
alpha is chosen to minimise FPR@95TPR on the validation set.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from methods.mahalanobis import fit_mahalanobis, compute_mahalanobis_scores
from methods.knn import fit_knn, compute_knn_scores
from metrics import compute_all_metrics


def _standardise(scores: np.ndarray, ref: np.ndarray):
    """Standardise scores using mean/std from ref split."""
    mu, sigma = ref.mean(), ref.std() + 1e-8
    return (scores - mu) / sigma


def fit_mahaknn(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    alphas: np.ndarray = None,
    k: int = 1,
):
    """Fit MahaKNN: compute Mahalanobis + KNN on val set, pick best alpha.

    Returns:
        class_means, precision  — for Mahalanobis
        train_feats             — for KNN
        alpha                   — best weight found on val
        val_is_ood              — ground truth on val set
    """
    if alphas is None:
        alphas = np.linspace(0, 1, 21)  # 0.0, 0.05, ..., 1.0

    # --- fit Mahalanobis on train ---
    class_means, precision = fit_mahalanobis(model, train_loader, num_classes, device)

    # --- fit KNN on train ---
    train_feats = fit_knn(model, train_loader, device)

    # --- score val set ---
    maha_val = compute_mahalanobis_scores(
        model, val_loader, class_means, precision, device
    ).numpy()
    knn_val = compute_knn_scores(
        model, val_loader, train_feats, device, k=k
    ).numpy()

    val_is_ood = np.array([
        int(batch["is_ood"].numpy().item())
        for batch in _iter_single(val_loader)
    ])

    # standardise on val
    maha_val_s = _standardise(maha_val, maha_val)
    knn_val_s  = _standardise(knn_val,  knn_val)

    # --- grid search alpha ---
    best_alpha, best_fpr = 0.5, float("inf")
    for alpha in alphas:
        combined = alpha * maha_val_s + (1 - alpha) * knn_val_s
        metrics = compute_all_metrics(combined, val_is_ood)
        if metrics["FPR@95TPR"] < best_fpr:
            best_fpr   = metrics["FPR@95TPR"]
            best_alpha = alpha

    print(f"  MahaKNN best alpha={best_alpha:.2f}  val FPR@95={best_fpr:.4f}")
    return class_means, precision, train_feats, best_alpha, maha_val, knn_val


def _iter_single(loader):
    """Yield one-sample-at-a-time from a loader (for collecting is_ood labels)."""
    for batch in loader:
        for i in range(len(batch["is_ood"])):
            yield {k: v[i:i+1] for k, v in batch.items() if isinstance(v, torch.Tensor)}


@torch.no_grad()
def compute_mahaknn_scores(
    model,
    test_loader: DataLoader,
    class_means: torch.Tensor,
    precision: torch.Tensor,
    train_feats: torch.Tensor,
    alpha: float,
    maha_val: np.ndarray,
    knn_val: np.ndarray,
    device: torch.device,
    k: int = 1,
) -> np.ndarray:
    """Compute MahaKNN OOD scores on test set.

    Scores are standardised using val-set statistics (no test leakage).
    """
    maha_test = compute_mahalanobis_scores(
        model, test_loader, class_means, precision, device
    ).numpy()
    knn_test = compute_knn_scores(
        model, test_loader, train_feats, device, k=k
    ).numpy()

    # standardise using val reference
    maha_test_s = _standardise(maha_test, maha_val)
    knn_test_s  = _standardise(knn_test,  knn_val)

    return alpha * maha_test_s + (1 - alpha) * knn_test_s
