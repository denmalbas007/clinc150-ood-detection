"""Mahalanobis Distance OOD detector — standard and layer-wise variants.

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_layer_features(
    model, loader: DataLoader, layer_idx: int, device: torch.device
):
    """Collect [CLS] hidden states from a specific BERT layer for in-domain samples."""
    model.eval()
    all_features, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids     = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"]
            is_ood         = batch["is_ood"]

            in_mask = ~is_ood
            if in_mask.sum() == 0:
                continue

            outputs = model.encoder(
                input_ids=input_ids[in_mask.to(device)],
                attention_mask=attention_mask[in_mask.to(device)],
                output_hidden_states=True,
            )
            # hidden_states: tuple of (num_layers+1) tensors, each (B, L, D)
            # index 0 = embedding layer, 1..12 = transformer layers
            cls_hidden = outputs.hidden_states[layer_idx][:, 0, :]  # (B, D)

            all_features.append(cls_hidden.cpu())
            all_labels.append(labels[in_mask])

    return torch.cat(all_features), torch.cat(all_labels)


def _fit_gaussian(features: torch.Tensor, labels: torch.Tensor, num_classes: int):
    """Fit per-class means and shared precision matrix."""
    hidden_dim = features.shape[1]

    class_means = torch.zeros(num_classes, hidden_dim)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_means[c] = features[mask].mean(0)

    centered = features - class_means[labels]
    cov = (centered.T @ centered) / (features.shape[0] - num_classes)
    cov += 1e-5 * torch.eye(hidden_dim)
    precision = torch.linalg.inv(cov)

    return class_means, precision


def _mahalanobis_scores_from_features(
    features: torch.Tensor,
    class_means: torch.Tensor,
    precision: torch.Tensor,
) -> torch.Tensor:
    """Compute min Mahalanobis distance over classes. (B,) → higher = more OOD."""
    diffs = features.unsqueeze(1) - class_means.unsqueeze(0)   # (B, C, D)
    md    = torch.einsum("bcd,de,bce->bc", diffs, precision, diffs)  # (B, C)
    return md.min(dim=-1).values


# ---------------------------------------------------------------------------
# Standard Mahalanobis (last layer, as in Podolskiy 2021)
# ---------------------------------------------------------------------------

def fit_mahalanobis(
    model, train_loader: DataLoader, num_classes: int, device: torch.device
):
    """Fit Gaussian on last-layer [CLS] features of in-domain training samples."""
    model.eval()
    all_features, all_labels = [], []

    with torch.no_grad():
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"]
            is_ood         = batch["is_ood"]

            in_mask = ~is_ood
            if in_mask.sum() == 0:
                continue

            features = model.get_features(
                input_ids[in_mask.to(device)], attention_mask[in_mask.to(device)]
            )
            all_features.append(features.cpu())
            all_labels.append(labels[in_mask])

    all_features = torch.cat(all_features)
    all_labels   = torch.cat(all_labels)
    return _fit_gaussian(all_features, all_labels, num_classes)


@torch.no_grad()
def compute_mahalanobis_scores(
    model,
    loader: DataLoader,
    class_means: torch.Tensor,
    precision: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute Mahalanobis OOD scores using last-layer features."""
    model.eval()
    class_means = class_means.to(device)
    precision   = precision.to(device)
    scores = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        features = model.get_features(input_ids, attention_mask)
        scores.append(
            _mahalanobis_scores_from_features(features, class_means, precision).cpu()
        )

    return torch.cat(scores)


# ---------------------------------------------------------------------------
# Layer-wise Mahalanobis analysis  (our novel contribution)
# ---------------------------------------------------------------------------

def layer_wise_mahalanobis(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    is_ood_gt,
    num_classes: int,
    device: torch.device,
    num_layers: int = 12,
):
    """Run Mahalanobis on every BERT layer (1..num_layers+1) and return metrics.

    Layer 0  = embedding layer
    Layer 1  = output of transformer block 1
    ...
    Layer 12 = output of transformer block 12  (= last layer, used by Podolskiy)

    Returns:
        layer_scores : dict {layer_idx -> np.ndarray of OOD scores on test set}
        layer_metrics: dict {layer_idx -> {'AUROC', 'FPR@95TPR', 'AUPR'}}
    """
    from metrics import compute_all_metrics

    model.eval()
    layer_scores  = {}
    layer_metrics = {}

    for layer_idx in range(1, num_layers + 1):   # 1..12 (BERT base: hidden_states[0..12])
        print(f"  Layer {layer_idx}/{num_layers} ...", end=" ")

        # --- fit ---
        train_feats, train_labels = _collect_layer_features(
            model, train_loader, layer_idx, device
        )
        class_means, precision = _fit_gaussian(train_feats, train_labels, num_classes)
        class_means = class_means.to(device)
        precision   = precision.to(device)

        # --- score test set ---
        scores = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                cls_hidden = outputs.hidden_states[layer_idx][:, 0, :]
                s = _mahalanobis_scores_from_features(cls_hidden, class_means, precision)
                scores.append(s.cpu())

        scores_np = torch.cat(scores).numpy()
        metrics   = compute_all_metrics(scores_np, is_ood_gt)

        layer_scores[layer_idx]  = scores_np
        layer_metrics[layer_idx] = metrics

        print(f"AUROC={metrics['AUROC']:.4f}  FPR@95={metrics['FPR@95TPR']:.4f}")

    return layer_scores, layer_metrics
