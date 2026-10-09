# MedConformalTriage

A research-oriented human-AI triage system for medical image classification.
Instead of forcing one prediction for every image, it produces a conformal
prediction set and routes each case to automatic processing, human review, or
rejection.

```text
image -> probabilities/logits -> prediction set -> OOD gate -> risk tier -> route
```

## Routing policy

- Singleton set, in-distribution, policy-compliant confidence: `auto`
- Multi-label set or a cautious high-risk case: `review`
- Empty set or detected OOD case: `reject`
- High-risk classes can use a smaller class-specific alpha and a stricter
  confidence threshold.

## Reported outcomes

- Auto-processing, human-review, and rejection rates
- Error rate among automatically processed cases
- Marginal, per-class, and per-site/device coverage
- Mean prediction-set size and worst-group coverage
- OOD rejection rate and high-risk-error capture rate

## Quick start

```bash
python -m pip install -e '.[dev]'
PYTHONPATH=src python examples/synthetic_demo.py
pytest -q
```

The demo writes JSON, CSV, and Markdown reports to `outputs/`.

## MedMNIST protocol

```bash
python -m pip install -e '.[medmnist]'
python examples/medmnist_protocol.py --dataset pathmnist
```

The official train split is reserved for model fitting, validation for
temperature/conformal/OOD calibration, and test for final reporting. Supported
entry points include PathMNIST, DermaMNIST, BloodMNIST, and OrganAMNIST.

## Scope and limitations

This is a research prototype, not a medical device. Marginal conformal coverage
does not imply coverage for every hospital, device, or demographic subgroup.
The report therefore exposes subgroup and worst-group coverage explicitly.
