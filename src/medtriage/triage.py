"""Human-AI routing policy."""
from dataclasses import dataclass
import numpy as np

AUTO, REVIEW, REJECT = "auto", "review", "reject"

@dataclass(frozen=True)
class TriageResult:
    prediction_sets: np.ndarray
    confidence: np.ndarray
    predicted_class: np.ndarray
    route: np.ndarray
    risk_tier: np.ndarray
    ood: np.ndarray

def route_cases(
    probabilities,
    prediction_sets,
    *,
    ood_scores=None,
    ood_threshold=None,
    high_risk_classes=(),
    high_risk_min_confidence: float = 0.9,
) -> TriageResult:
    p = np.asarray(probabilities, float)
    sets = np.asarray(prediction_sets, bool)
    pred = p.argmax(axis=1); confidence = p.max(axis=1); size = sets.sum(axis=1)
    ood = np.zeros(len(p), dtype=bool) if ood_scores is None else np.asarray(ood_scores) > ood_threshold
    route = np.full(len(p), REVIEW, dtype=object)
    route[(size == 1) & ~ood] = AUTO
    route[(size == 0) | ood] = REJECT
    high_risk = np.isin(pred, list(high_risk_classes))
    route[high_risk & (confidence < high_risk_min_confidence) & ~ood] = REVIEW
    tier = np.full(len(p), "medium", dtype=object)
    tier[route == AUTO] = "low"; tier[route == REJECT] = "critical"; tier[high_risk] = "high"
    return TriageResult(sets, confidence, pred, route, tier, ood)
