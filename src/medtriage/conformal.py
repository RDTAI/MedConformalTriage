"""Split and class-conditional conformal prediction for multiclass models."""
from __future__ import annotations
import numpy as np

def _quantile(scores: np.ndarray, alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    n = len(scores)
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return float("inf") if rank > n else float(np.sort(scores)[rank - 1])

def fit_class_conditional(
    calibration_probabilities,
    calibration_labels,
    *,
    alpha: float = 0.1,
    alpha_by_class: dict[int, float] | None = None,
) -> np.ndarray:
    """Return one finite-sample corrected threshold per class."""
    p = np.asarray(calibration_probabilities, float)
    y = np.asarray(calibration_labels, int).reshape(-1)
    if p.ndim != 2 or len(p) != len(y):
        raise ValueError("calibration arrays have incompatible shapes")
    thresholds = np.empty(p.shape[1])
    for label in range(p.shape[1]):
        class_alpha = (alpha_by_class or {}).get(label, alpha)
        scores = 1 - p[y == label, label]
        thresholds[label] = _quantile(scores, class_alpha)
    return thresholds

def prediction_sets(probabilities, thresholds) -> np.ndarray:
    p = np.asarray(probabilities, float)
    q = np.asarray(thresholds, float)
    if p.ndim != 2 or q.shape != (p.shape[1],):
        raise ValueError("threshold count must match number of classes")
    return (1 - p) <= q[None, :]
