# Full model-validation coverage follow-up

2026-10-21 ICT. This is post-screen development coverage, not independent confirmation. The original 142-QA primary screen and its inconclusive decision are unchanged. Three fixed seed42 LAST296 checkpoints were evaluated on all sealed fold0 MODEL_VAL: 1,432 QA, 321 images, 314 components. No fitting, optimizer updates, calibration/test inference, threshold tuning, or checkpoint selection occurred.

| Fixed checkpoint | Correct / 1,432 | Accuracy | OPEN correct / 628 | OPEN accuracy | Canonical fracture AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Raw B | 1,128 | 78.77% | 484 | 77.07% | 0.882709 | 0.662038 |
| Sequence uniform | 1,143 | 79.82% | 503 | 80.10% | 0.882974 | 0.657117 |
| Sequence rarity | 1,130 | 78.91% | 484 | 77.07% | 0.883240 | 0.664167 |

All three have aggregate TP53/FN4/TN177/FP87 at fixed raw threshold0.5; matching aggregate counts do not imply identical image-level predictions. Clinical paired intervals include zero. Uniform has a descriptive +1.05 percentage-point overall and +3.03-point OPEN difference versus B, not an established independent clinical gain.

Predeclared full-cohort bootstrap contrast is rarity minus controls (500 paired component resamples). Rarity minus B overall 95% interval [-0.007983, 0.011175], OPEN [-0.020912, 0.019626]; both include zero. Rarity minus uniform OPEN interval [-0.047824, -0.014446] favors uniform on this inspected development cohort. Do not reinterpret it as a preregistered uniform-minus-B confirmation or replace the original 142-QA primary endpoint. Repeated inspection and a single training seed limit inference.

TRAIN-answer-frequency coverage (computed after prediction, not supplied to the model): all three fail all 20 OPEN rows whose normalized answers were unseen in pilot TRAIN. Among 45 seen-rare OPEN rows, B/uniform/rarity get 15/21/17 correct. These are descriptive strata, not clinical equivalence judgments or permission to add validation answers to TRAIN, vocabulary, weights, or templates.

Public inference used original images and raw questions only, with T/V/L/P off and no annotation ROI, GT type/region, loss adapter, Platt calibration, or threshold fitting. Fresh-process seed90211 exactly reproduced all three first-pass seed90210 prediction/probability/metric/coverage records. Prior canonical321 batching changed no answers; maximum probability differences B/uniform/rarity were 2.8461e-6 / 5.9605e-8 / 5.9605e-8, retained rather than selected away.

Measured evaluation plus fresh-process replay and bootstrap: 1,717.191334 seconds (28 minutes 37.191 seconds), excluding the separate no-model artifact audit. First passes: 266.925 / 263.177 / 268.774 seconds. This is evaluation timing, not remaining total-step3 time.

Report: `full_modelval_coverage_v3_1/report.json`, SHA256 `5353a6a9c56d86988ee2c4419232082ccb08e5e2859d0e0e4a8fdf37218dd6b3`. Protocol SHA256 `aed29dfe5765f773a3f953317179dd0acfefe306d8ccb96381413002550b2448`. Separate no-model artifact audit PASSED: all bound hashes, 1,432 row identities, canonical321 subset, metrics, coverage and 500 component bootstraps recomputed. See `full_modelval_coverage_v3_1/independent_artifact_audit.json`. Original 190-artifact technical seal also verified unchanged.

Decision: raw B remains provisional; do not promote a final solution or launch longer training from these points. Next work must close remaining data/provenance scope and define any matched replication/confirmation prospectively, without chasing this repeatedly inspected validation set. Existing FracAtlas test is not untouched evidence. Full step3, honest confirmation, calibration, final contributions, Final Solution and production release remain incomplete. Historical and failed artifacts are preserved.
