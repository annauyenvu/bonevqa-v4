# Outcome: controlled loss screen

Verified 2026-10-21 00:54 ICT. Both predeclared pilots and the independent evaluation are complete. This does not complete all of step 3 or establish a production model.

| LAST296, training seed42 | Correct /142 QA | Correct /64 OPEN | AUROC /321 | AP /321 |
|---|---:|---:|---:|---:|
| Existing raw token-mean B | 113 | 50 | 0.882709 | 0.662038 |
| Uniform per-sequence loss | 112 | 50 | 0.882974 | 0.657117 |
| TRAIN rarity-weighted per-sequence loss | 114 | 51 | 0.883240 | 0.664167 |

The rarity arm is one answer ahead of raw B, not an established improvement. Paired500-resample component bootstrap rarity-minus-B QA accuracy95% interval is [-2.23,+3.82] percentage points; OPEN [-5.68,+8.41] points. Clinical AUROC difference interval [-0.004208,+0.005770] and AP [-0.015848,+0.020470] cross zero. Against the uniform control, QA overall andOPEN intervals include zero, and clinicalAUROC/AP intervals cross zero. All three have identical aggregate clinical confusion counts at fixedraw0.5: TP53,FN4,TN177,FP87; sensitivity92.98%,specificity67.05%. Identical aggregate counts do not mean every per-image decision is identical. Clinical321 contains no OPEN questions; its OPEN0 fields are not a measured OPEN task.

Neither loss candidate is promoted; no long training or threshold/epoch/seed/weight/template search follows from this screen. Passing gradients and finite updates certify technical execution, not benefit. TRAIN rare-answer errors motivated the hypothesis, but these aggregate development scores do not establish a specific rare-answer improvement or causal imbalance.

Both arms started from the same263cleanROCO PREtask tensor values, seed42,1173TRAIN/142model-validation QA, same component-aware schedules,4epochs296updates, batch4accum4, offlineCPU8float32. Uniform is the matched control for token-to-sequence reduction versus rarity weighting. Original rawimages/questions/routing and all classification/fracture/type/contrastive terms unchanged; no V/L/P/T, annotationROI, routing adapter or new labels. TRAIN-only frequency weights never enter public inference. Interrupted v3_2 is preserved and excluded from quality selection; recoveryv3_3 restarted cleanly rather than resuming partial weights. Uniform took1224.578sec; rarity1180.868sec.

Actual-Qwen CE/selected-gradient equivalence, finite nonzero answer gradients/disposable updates, clean263initialization and public-inference/evalbypass gates passed before launch. Both completed LAST/optimizer states finite at296, histories4epochs142QA/zero skippedbatches. Per-arm TRAINforward1176/rows4692/OPEN2136/nativeeval568 counters audited. The four gate smokeupdates were discarded.

Independent evaluation took231.763sec including a fresh90211-constructor subprocess replay of90210 first pass. Every saved answer, probability and metric matched exactly; weights unchanged. Public inputs were only originalimages/rawquestions, adapterOFF; threshold0.5,noPlatt,no calibration/test. A separate no-model artifact auditor rehashed sources/manifests/checkpoints/predictions/replay, recomputed all record identities/native-history metrics, full321confusions and all500seed20261020 component bootstraps exactly. QA32components, clinical314components. Original190technical source/config/test artifacts unchanged.

Evidence: `B_sequence_loss_quality_v3_2/report.json` SHA a86fde874773a9e572d89bf28400e9e90e9c0693fbed944bca7ffef12c372ac6; `fresh_process_replay.json`; `independent_artifact_audit.json`; execution `B_sequence_loss_pilots_v3_3/`. Uniform LAST SHA8a4dbbb7abf60ad55e4e0197063743c709a5f135b501c5e5e2c6cecb4328e009; rarity LAST SHA2c818267913ce69e66c30803eef67553aba0f2b77d81aa97b1f994f53b5d9ec1.

Limitations: one training seed, small development cohort and repeated model-validation inspection; no independent confirmation. Existing FracAtlas test is not untouched. PMC contaminated historical checkpoints/test remain excluded. T/V/L/P efficacy remains unestablished; raw B is provisional, not Final Solution. Honest predeclared component-nested/verified-independent confirmation, outstanding data/provenance decisions and reserved-role calibration remain unresolved. Do not translate this completed bounded screen into completion of the seven-step plan.
