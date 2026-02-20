"""OOD detection evaluation metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score


def auroc(ood_scores: np.ndarray, is_ood: np.ndarray) -> float:
    """AUROC: treats OOD as positive class.

    Args:
        ood_scores: higher score = more likely OOD
        is_ood:     binary ground truth (1 = OOD, 0 = in-domain)
    """
    return roc_auc_score(is_ood, ood_scores)


def fpr_at_tpr(ood_scores: np.ndarray, is_ood: np.ndarray, tpr_threshold: float = 0.95) -> float:
    """FPR at a given TPR level (default: FPR@95TPR)."""
    fpr_arr, tpr_arr, _ = roc_curve(is_ood, ood_scores)
    # Find lowest FPR where TPR >= threshold
    idx = np.searchsorted(tpr_arr, tpr_threshold)
    if idx >= len(fpr_arr):
        return 1.0
    return float(fpr_arr[idx])


def aupr(ood_scores: np.ndarray, is_ood: np.ndarray) -> float:
    """AUPR: Average Precision (OOD as positive class)."""
    return average_precision_score(is_ood, ood_scores)


def compute_all_metrics(ood_scores: np.ndarray, is_ood: np.ndarray) -> dict:
    """Compute AUROC, FPR@95TPR, AUPR in one call."""
    return {
        "AUROC": auroc(ood_scores, is_ood),
        "FPR@95TPR": fpr_at_tpr(ood_scores, is_ood, 0.95),
        "AUPR": aupr(ood_scores, is_ood),
    }
