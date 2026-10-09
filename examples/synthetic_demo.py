"""End-to-end deterministic demo; no medical data are downloaded."""
from pathlib import Path
import numpy as np
from medtriage import *

rng = np.random.default_rng(7); classes = 4
cal_y = rng.integers(classes, size=800)
cal_logits = rng.normal(size=(800, classes)); cal_logits[np.arange(800), cal_y] += 2.2
test_y = rng.integers(classes, size=1000)
test_logits = rng.normal(size=(1000, classes)); test_logits[np.arange(1000), test_y] += 1.9
softmax = lambda z: np.exp(z-z.max(1, keepdims=True))/np.exp(z-z.max(1, keepdims=True)).sum(1, keepdims=True)
cal_p, test_p = softmax(cal_logits), softmax(test_logits)
thresholds = fit_class_conditional(cal_p, cal_y, alpha=.1, alpha_by_class={3: .05})
sets = prediction_sets(test_p, thresholds)
id_energy = energy_score(cal_logits); test_energy = energy_score(test_logits)
ood_logits = rng.normal(-.2, .3, size=(80, classes)); all_p=np.vstack([test_p, softmax(ood_logits)])
all_sets=np.vstack([sets, prediction_sets(softmax(ood_logits), thresholds)])
all_scores=np.r_[test_energy, energy_score(ood_logits)]; all_y=np.r_[test_y, rng.integers(classes,size=80)]
groups=np.array(["site_a" if i%2 else "site_b" for i in range(len(all_y))])
result=route_cases(all_p, all_sets, ood_scores=all_scores, ood_threshold=fit_ood_threshold(id_energy), high_risk_classes={3}, high_risk_min_confidence=.95)
id_mask=np.r_[np.ones(len(test_y),bool),np.zeros(len(ood_logits),bool)]
report=evaluate_triage(result, all_y, groups=groups, high_risk_classes={3}, in_distribution_mask=id_mask)
Path("outputs").mkdir(exist_ok=True); write_outputs(report,"outputs/synthetic_triage")
print(render_markdown(report))
