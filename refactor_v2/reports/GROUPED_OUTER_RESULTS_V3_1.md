# Fixed native B grouped outer evaluation

Verified result, 22 October 2026 (ICT): the fixed raw pipeline achieves 75.26% canonical fracture balanced accuracy. Sensitivity is 55.94% despite overall VQA exact accuracy of 90.00%. Frozen per-fold calibration improves probability calibration but reduces the primary classification endpoint. These results do not establish clinical readiness or architectural superiority.

## Frozen experiment and evidence

The experiment uses seed 42, native B, two training epochs and fixed LAST checkpoints, original images and raw public questions. T/V/L/P are disabled. Inference receives only `images` and `questions`; existing answers, answer types and annotations remain evaluation metadata. There is no annotation ROI input or invented clinical label. This experiment does not evaluate LaPA GM/LF contributions and provides no PF graph evidence.

The four new folds contain 5,768 optimizer updates and took 25,588 seconds (7 h 6 min) in worker execution. Historical native fold 0 is reused with disclosure. Each new fold passed exact schedule/count/checkpoint artifact checks. Startup VisualLoRA gradient/update QC passed 24/24; real-model public-core integration checked all saved trainable tensors unchanged during inference. This is not a claim that every module was independently measured on every training update.

Within each fold, train, model-validation, calibration and outer roles are disjoint by known component. The five outer roles cover exactly once 4,013 images, 3,920 known components and 17,998 QA records. All five prediction/fit outputs replay exactly in fresh processes. Saved labels/order, fixed calibration refits, raw/calibrated probabilities and clinical metrics passed independent artifact audits.

The matched comparison uses the same fixed model, outer images/questions and labels; only the previously specified per-fold calibration transform differs. Calibration is LogisticRegression with C=1, lbfgs, max_iter=1000, tol=1e-8, seed42, intercept enabled, no class weighting, applied to clipped-probability logits from canonical calibration records. Both decision thresholds are fixed at 0.5. There is no model, epoch, sensitivity target or threshold search from outer results.

## Results

| Endpoint | Raw | Calibrated |
| --- | ---: | ---: |
| Canonical fracture balanced accuracy | 75.26% | 73.52% |
| Sensitivity | 55.94% | 50.35% |
| Specificity | 94.57% | 96.69% |
| AUROC | 0.88399 | 0.88028 |
| Average precision | 0.71627 | 0.70479 |
| Brier score (lower better) | 0.09562 | 0.08891 |
| ECE10 (lower better) | 0.05728 | 0.02875 |
| Overall VQA exact accuracy | 90.00% | 90.17% |
| CLOSED exact accuracy | 90.91% | 91.21% |
| OPEN exact accuracy | 88.85% | 88.85% |
| TP / FN / TN / FP | 400 / 315 / 3119 / 179 | 360 / 355 / 3189 / 109 |

| Fold | Raw fracture BA | Calibrated fracture BA |
| --- | ---: | ---: |
| 0 (historical reuse) | 72.66% | 71.50% |
| 1 | 74.27% | 71.49% |
| 2 | 79.24% | 76.27% |
| 3 | 73.66% | 73.25% |
| 4 | 76.46% | 75.10% |

The primary raw BA bootstrap interval is **73.16–77.05%**. Calibrated BA is **71.44–75.38%**. The paired calibrated-minus-raw BA difference is **−1.74 percentage points**, interval **−2.65 to −0.91 points**. All 2,000 primary draws are valid. These are percentile intervals from paired component resampling within each outer fold, seed 20261021, holding models and fitted calibrators fixed. No refitting occurs within a bootstrap draw. The intervals are conditional descriptive uncertainty; they do not cover training seed variability, unknown patient grouping or multiple-endpoint inferential selection.

The calibration comparison trades 40 additional false negatives for 70 fewer false positives. Improved Brier/ECE and overall QA accuracy therefore do not demonstrate improved fracture detection. Retain raw results as the frozen primary endpoint; do not promote calibrated decisions on the basis of the small overall QA gain. No production candidate or final architecture contribution is established.

## Limits and remaining closure

All FracAtlas folds are development-informed; historical test data had been inspected. Known components are not verified patient identities. This is an honest grouped evaluation, not untouched independent clinical confirmation. Contaminated historical PMC `train_lon` checkpoints/test are not used as independent evidence. This result neither repairs that contamination nor establishes a benefit from PMC pretraining.

The grouped training/evaluation experiment and its replay/bootstrap audits are complete. Full step 3 remains subject to the previously approved plan's exit review and report reconciliation; this report does not mark the seven-step plan complete. Further tuning on these outer results would constitute a new development experiment, requiring explicit disclosure and a newly frozen evaluation design. No additional training or threshold search is launched here.

## Verified artifacts

- [Five-fold bindings and role audit](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/grouped_evaluation_bindings_v3_1/independent_audit.json)
- [Inference/replay runtime and status](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/grouped_reserved_evaluation_v3_1/report.json): 4,701.25 seconds, 78 min 21 sec.
- Per-fold independent audits: `grouped_reserved_evaluation_v3_1/fold_0_independent_audit.json` through `fold_4_independent_audit.json`.
- [All pooled/per-fold metrics, 2,000 draw values and intervals](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/grouped_outer_bootstrap_v3_1/results.json)
- [Analysis replay report](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/grouped_outer_bootstrap_v3_1/report.json): 49.36 seconds including fresh-process replay.
- [Independent primary bootstrap and all-percentile audit](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/grouped_outer_bootstrap_v3_1/independent_audit.json)

Audit scope: all 2,000 paired sensitivity/specificity/BA draws were independently reconstructed from per-component confusion counts; all stored percentiles and undefined counts were independently checked. Other metric points were independently checked in per-fold audits and all full analysis outputs replayed exactly; there is no claim that every ranking-metric draw was independently recomputed by a second algorithm.
