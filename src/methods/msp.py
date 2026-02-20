"""Maximum Softmax Probability (MSP) OOD detector.

Reference:
    Hendrycks & Gimpel (2017). A Baseline for Detecting Misclassified and
    Out-of-Distribution Examples in Neural Networks. ICLR 2017.
    https://arxiv.org/abs/1610.02136
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_msp_scores(model, loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Compute MSP OOD scores.

    Lower score → more likely OOD (score = max softmax prob, so we negate it).
    Returns negative max-softmax so that higher value = more OOD (consistent
    with other detectors).
    """
    model.eval()
    scores = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        msp = probs.max(dim=-1).values
        scores.append(-msp.cpu())  # negate: higher = more OOD

    return torch.cat(scores)
