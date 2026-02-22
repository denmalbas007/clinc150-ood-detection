"""Per-Class k-Nearest Neighbors OOD detector.

Our method. Extension of Sun et al. (2022) where neighbours are retrieved
only from the predicted class subset of the training set, rather than the
full training bank. This gives a tighter, class-specific decision boundary:
an utterance is flagged as OOD if it is far from its own predicted class
cluster, not merely far from the global nearest neighbour.

Intuition:
    Standard KNN uses the global training bank. An OOD sample may happen
    to land near some in-domain class other than the predicted one, yielding
    a low (in-domain) OOD score. Per-class KNN eliminates this confound by
    measuring distance only within the predicted class, making it harder for
    OOD samples to "hide" behind irrelevant neighbours.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def fit_per_class_knn(
    model,
    train_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    normalize: bool = True,
):
    """Collect per-class normalised training embeddings.

    Returns:
        class_banks: list of length num_classes, each element is a
                     (N_c, D) tensor of normalised embeddings for class c.
    """
    model.eval()
    class_banks = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for batch in train_loader:
            input_ids     = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels        = batch["label"]          # (B,)  -1 for OOD
            is_ood        = batch["is_ood"]

            in_mask = ~is_ood
            if in_mask.sum() == 0:
                continue

            feats = model.get_features(
                input_ids[in_mask.to(device)],
                attention_mask[in_mask.to(device)],
            )                                       # (B_in, D)
            if normalize:
                feats = F.normalize(feats, dim=-1)

            labels_in = labels[in_mask]             # (B_in,)
            for feat, lbl in zip(feats.cpu(), labels_in):
                class_banks[int(lbl)].append(feat)

    # Stack each class into a tensor; handle empty classes gracefully
    class_banks = [
        torch.stack(b) if b else torch.zeros(0, 768)
        for b in class_banks
    ]
    return class_banks                              # list[(N_c, D)]


@torch.no_grad()
def compute_per_class_knn_scores(
    model,
    loader: DataLoader,
    class_banks: list,
    device: torch.device,
    k: int = 1,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute per-class KNN OOD score (higher = more OOD).

    For each test sample:
      1. Get predicted class c = argmax(logits).
      2. Retrieve top-k cosine similarities within class_banks[c].
      3. OOD score = negative mean top-k similarity.

    If a class bank is empty (edge case), falls back to score = 0.
    """
    model.eval()
    # Pre-move all class banks to device
    class_banks_dev = [b.to(device) for b in class_banks]
    scores = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits, feats = model(input_ids, attention_mask, return_features=True)   # (B, C), (B, D)
        if normalize:
            feats = F.normalize(feats, dim=-1)

        preds = logits.argmax(dim=-1)              # (B,)

        batch_scores = []
        for feat, pred in zip(feats, preds):
            bank = class_banks_dev[int(pred)]      # (N_c, D)
            if bank.shape[0] == 0:
                batch_scores.append(torch.tensor(0.0))
                continue
            sim = feat @ bank.T                    # (N_c,)
            actual_k = min(k, sim.shape[0])
            topk_sim = sim.topk(actual_k).values.mean()
            batch_scores.append(-topk_sim.cpu())

        scores.append(torch.stack(batch_scores))

    return torch.cat(scores)
