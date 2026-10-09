"""Safety and workload metrics for conformal triage."""
import numpy as np
from .triage import AUTO, REVIEW, REJECT, TriageResult

def _coverage(sets, labels):
    y = np.asarray(labels, int)
    return float(sets[np.arange(len(y)), y].mean()) if len(y) else float("nan")

def evaluate_triage(result: TriageResult, labels, *, groups=None, high_risk_classes=(), in_distribution_mask=None):
    y = np.asarray(labels, int); correct = result.predicted_class == y
    id_mask = np.ones(len(y), dtype=bool) if in_distribution_mask is None else np.asarray(in_distribution_mask, bool)
    report = {
        "n": int(len(y)),
        "coverage": _coverage(result.prediction_sets[id_mask], y[id_mask]),
        "mean_set_size": float(result.prediction_sets[id_mask].sum(1).mean()),
        "auto_rate": float((result.route == AUTO).mean()),
        "review_rate": float((result.route == REVIEW).mean()),
        "reject_rate": float((result.route == REJECT).mean()),
        "auto_error_rate": float((~correct[result.route == AUTO]).mean()) if np.any(result.route == AUTO) else None,
        "ood_rejection_rate": float((result.route[result.ood] == REJECT).mean()) if result.ood.any() else None,
    }
    report["class_coverage"] = {
        str(c): _coverage(result.prediction_sets[id_mask & (y == c)], y[id_mask & (y == c)]) for c in np.unique(y[id_mask])
    }
    high = np.isin(y, list(high_risk_classes)) & ~correct
    report["high_risk_error_capture"] = float((result.route[high] != AUTO).mean()) if high.any() else None
    if groups is not None:
        g = np.asarray(groups)
        report["group_coverage"] = {str(v): _coverage(result.prediction_sets[id_mask & (g == v)], y[id_mask & (g == v)]) for v in np.unique(g[id_mask])}
        report["worst_group_coverage"] = min(report["group_coverage"].values())
    return report
