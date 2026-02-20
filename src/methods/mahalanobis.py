"""Mahalanobis Distance OOD detector.

References:
    Lee et al. (2018). A Simple Unified Framework for Detecting
    Out-of-Distribution Samples and Adversarial Attacks. NeurIPS 2018.
    https://arxiv.org/abs/1807.03888

    Podolskiy et al. (2021). Revisiting Mahalanobis Distance for
    Transformer-Based Out-of-Domain Detection. AAAI 2021.
    https://arxiv.org/abs/2101.03778
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def fit_mahalanobis(
    model, train_loader: DataLoader, num_classes: int, device: torch.device
):
    """Compute per-class means and shared covariance from training features.

    Returns:
        class_means: (num_classes, hidden_dim)
        precision:   (hidden_dim, hidden_dim)  — inverse of shared covariance
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            is_ood = batch["is_ood"]

            # Only in-domain samples for fitting
            in_mask = ~is_ood
            if in_mask.sum() == 0:
                continue

            features = model.get_features(
                input_ids[in_mask.to(device)], attention_mask[in_mask.to(device)]
            )
            all_features.append(features.cpu())
            all_labels.append(labels[in_mask])

    all_features = torch.cat(all_features)   # (N, D)
    all_labels = torch.cat(all_labels)        # (N,)
    hidden_dim = all_features.shape[1]

    # Per-class means
    class_means = torch.zeros(num_classes, hidden_dim)
    class_counts = torch.zeros(num_classes)
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            class_means[c] = all_features[mask].mean(0)
            class_counts[c] = mask.sum().float()

    # Shared tied covariance
    centered = all_features - class_means[all_labels]
    cov = (centered.T @ centered) / (all_features.shape[0] - num_classes)
    # Regularise for numerical stability
    cov += 1e-5 * torch.eye(hidden_dim)
    precision = torch.linalg.inv(cov)

    return class_means, precision


@torch.no_grad()
def compute_mahalanobis_scores(
    model,
    loader: DataLoader,
    class_means: torch.Tensor,
    precision: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute Mahalanobis OOD scores (min distance over classes).

    Higher score → more likely OOD.
    """
    model.eval()
    class_means = class_means.to(device)
    precision = precision.to(device)
    scores = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        features = model.get_features(input_ids, attention_mask)  # (B, D)

        # Mahalanobis distance to each class: (B, C)
        diffs = features.unsqueeze(1) - class_means.unsqueeze(0)  # (B, C, D)
        # d^2 = diff @ Sigma^{-1} @ diff^T
        md = torch.einsum("bcd,de,bce->bc", diffs, precision, diffs)
        # Min over classes — score: smaller = more in-domain
        min_dist = md.min(dim=-1).values
        scores.append(min_dist.cpu())

    return torch.cat(scores)
