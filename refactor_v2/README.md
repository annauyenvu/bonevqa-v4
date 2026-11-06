# BoneVQA refactor v2

This directory is the development implementation for protocol, routing, and
gradient/overfit repairs. Historical experiment snapshots under
`exp4_multiseed/` and `mandatory_experiments_20261011/` remain immutable so
their hashes and reported results stay reproducible.

The code in `src/` starts from the reconstructed runtime used by the audit and
is intentionally versioned separately until the new protocol passes the gates
in `tests/`.

## Re-run the gates

Use the experiment virtual environment:

```bash
PYTHONPATH=refactor_v2 /Users/uyen/Downloads/BoneMedVQA/.venv/bin/python -m pytest -q refactor_v2/tests
PYTHONPATH=refactor_v2 /Users/uyen/Downloads/BoneMedVQA/.venv/bin/python refactor_v2/scripts/run_step3_gates.py
PYTHONPATH=refactor_v2 /Users/uyen/Downloads/BoneMedVQA/.venv/bin/python refactor_v2/scripts/run_full_model_overfit.py
```

Production training must set `BONEVQA_DATA` to the directory containing the
sealed `fracatlas_component_clean` manifests. The reference configuration is
`configs/fracatlas_component_clean_v2.yaml`.

Detector retraining must use the same component split:

```bash
PYTHONPATH=refactor_v2 /Users/uyen/Downloads/BoneMedVQA/.venv/bin/python \
  refactor_v2/scripts/train_detector_v2.py \
  --split_manifest .evaluation_runtime/data/fracatlas_component_clean/component_split_manifest.json
```

The API fails at startup when `BONEVQA_CKPT` is absent or incompatible. A mock
engine is available only when `BONEVQA_ALLOW_MOCK=1` is explicitly set for UI
development.
