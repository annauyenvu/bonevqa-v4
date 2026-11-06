# Five fold-specific PREtask initializations

2026-10-21 06:55 ICT. Five zero-QA initial checkpoints are exported under `perfold_PREtask_roundtrip_v3_1/`, one per existing fold. These are **initial states, not trained models**. Total1,267,404,725 bytes retained for traceability/future gated use. No optimizer was created or stepped; no reserved-role model inference occurred.

| Fold | TRAIN vocabulary answers | Canonical TRAIN no / yes | TRAIN-derived positive weight |
|---|---:|---:|---:|
| 0 | 125 | 2110 / 458 | 4.606986900 |
| 1 | 117 | 2110 / 458 | 4.606986900 |
| 2 | 115 | 2110 / 458 | 4.606986900 |
| 3 | 127 | 2111 / 458 | 4.609170306 |
| 4 | 120 | 2110 / 458 | 4.606986900 |

Each starts from verified newROCO BEST plus seed42 task-head initialization; only its own TRAIN vocabulary and class counts are used. All255 non-task-head tensor hashes match across folds. All263 fold0 tensors also match the previous125-vocabulary loss gate. Different vocabulary/task-head shapes across folds are expected; objective controls within a fold must consume the same exported checkpoint SHA rather than independently initialize heads. This does not claim any matched trained-arm result.

A fresh process with constructor seed90211 loaded each checkpoint through the frozen strict manifest/trainable-state loader. All263 tensors and all4 public TRAIN inference records per fold exactly match the export process,20 records total. Public inputs are original images/raw questions only, no GT type/region/ROI. Answer correctness is not an endpoint in this initialization check. No accuracy or clinical improvement is inferred.

Original driver exported all5 files, then its child aborted at OpenMP shared-memory startup withSIGABRT6/Can't open SHM. Original failed progress, source and failure record remain unchanged. Only the unchanged replay function was separately run with appropriate shared-memory process permission, same offlineCPU8 runtime, and passed. Exact OS root cause was not exhaustively isolated. No duplicate export, model/data edit, package change or network access occurred. RECOVERY_POINTER.json identifies current evidence; do not rerun export because the preserved original progress still says failed.

First no-model auditor also stopped at a config comparison: saved SAM config includes frozen default null fields absent from the candidate's partial nested dict. That failed source/record are preserved. Separate corrected auditor reproduces the exact frozen default/nested-dict merge and requires complete resolved-config equality, not a whitelist. It then verifies all5 physical checkpoint hashes/manifests/263 tensor hashes,fold vocabularies/TRAIN bindings,source/upstream hashes and fresh-process assertions. No neural model was instantiated by the artifact auditor. Current successful evidence is `perfold_PREtask_roundtrip_recovery_v3_3.json`, not a fabricated successful original-driver report.

Original driver failed before recording full elapsed time; no end-to-end runtime estimate is invented. Artifact audit3.364seconds is not a model-training rate. A zero-QA reload does not certify optimizer/RNG training resume or every per-fold post-update behavior. Original190 technical artifacts hash-verify unchanged. No data-manifest changes, quality training, clinical calibration or outer evaluation. Prior1432accuracy/clinical conclusions unchanged; fullstep3,FinalSolution and production remain incomplete. Exact hashes/remaining protocol and execution gates are in CURRENT_GATE_LEDGER_V3_19.json.
