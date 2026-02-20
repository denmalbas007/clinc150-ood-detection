"""k-Nearest Neighbors OOD detector.

Reference:
    Sun et al. (2022). Out-of-Distribution Detection with Deep Nearest Neighbors.
    ICML 2022. https://arxiv.org/abs/2204.06507
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def fit_knn(
    model, train_loader: DataLoader, device: torch.device, normalize: bool = True
):
    """Collect normalised training embeddings for KNN lookup."""
    model.eval()
    all_features = []

    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            is_ood = batch["is_ood"]
            in_mask = ~is_ood
            if in_mask.sum() == 0:
                continue
            feats = model.get_features(
                input_ids[in_mask.to(device)], attention_mask[in_mask.to(device)]
            )
            if normalize:
                feats = F.normalize(feats, dim=-1)
            all_features.append(feats.cpu())

    return torch.cat(all_features)  # (N_train, D)


@torch.no_grad()
def compute_knn_scores(
    model,
    loader: DataLoader,
    train_features: torch.Tensor,
    device: torch.device,
    k: int = 1,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute negative kNN distance as OOD score (higher = more OOD)."""
    model.eval()
    train_features = train_features.to(device)  # (N, D)
    scores = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        feats = model.get_features(input_ids, attention_mask)  # (B, D)
        if normalize:
            feats = F.normalize(feats, dim=-1)

        # Cosine similarity → (B, N)
        sim = feats @ train_features.T
        # kNN: mean of top-k similarities (higher sim = more in-domain)
        topk_sim = sim.topk(k, dim=-1).values.mean(dim=-1)
        # Negate: higher score = more OOD
        scores.append(-topk_sim.cpu())

    return torch.cat(scores)
