"""Monte Carlo Dropout OOD detector.

Reference:
    Gal & Ghahramani (2016). Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning. ICML 2016.
    https://arxiv.org/abs/1506.02142
"""
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_mc_dropout_scores(
    model,
    loader: DataLoader,
    device: torch.device,
    n_passes: int = 20,
) -> torch.Tensor:
    """Compute MC Dropout OOD scores (predictive entropy).

    Higher entropy → more uncertain → more likely OOD.
    """
    scores = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mean_probs = model.forward_mc(input_ids, attention_mask, n_passes=n_passes)

        # Predictive entropy: H[p] = -sum_y p_y * log(p_y)
        eps = 1e-8
        entropy = -(mean_probs * (mean_probs + eps).log()).sum(dim=-1)
        scores.append(entropy.cpu())

    return torch.cat(scores)
