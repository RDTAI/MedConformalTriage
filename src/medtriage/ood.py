"""OOD scores and calibration-set thresholding."""
import numpy as np

def energy_score(logits, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, float) / temperature
    maximum = z.max(axis=1, keepdims=True)
    return -temperature * (maximum[:, 0] + np.log(np.exp(z - maximum).sum(axis=1)))

def fit_ood_threshold(in_distribution_scores, false_positive_rate: float = 0.05) -> float:
    """Lower energy is more ID-like; reject scores above this quantile."""
    return float(np.quantile(np.asarray(in_distribution_scores, float), 1 - false_positive_rate))
