"""Energy-based OOD detector.

Reference:
    Liu et al. (2020). Energy-based Out-of-distribution Detection. NeurIPS 2020.
    https://arxiv.org/abs/2010.03759
"""
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_energy_scores(
    model, loader: DataLoader, device: torch.device, temperature: float = 1.0
) -> torch.Tensor:
    """Compute energy-based OOD scores.

    E(x) = -T * log(sum_y exp(f_y(x) / T))
    Higher energy → more likely OOD.
    """
    model.eval()
    scores = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        energy = -temperature * torch.logsumexp(logits / temperature, dim=-1)
        scores.append(energy.cpu())

    return torch.cat(scores)
