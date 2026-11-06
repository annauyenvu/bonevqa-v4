# Bounded step4 execution started — 23 October 2026 ICT

Actual runner smoke passed for B, V-real and V-placebo: two disposable optimizer updates per arm, selected finite nonzero fracture/local gradients and parameter changes, followed by exact eight-record prediction/probability replay in separate processes. The smoke deliberately includes proposal-present TRAIN batches in both accumulation groups to exercise the initially gated projection; the unchanged full quality schedule includes empty-proposal images. Failed sparse-smoke and sandbox/config-comparison attempts are preserved in step4_localization_v3_1.

execution_release.json now binds the validated dependencies and releases only the three fixed seed42 four-epoch arms,292updates each. The sequential operation driver has begun B, followed by V-real and V-placebo, with exclusive locks and no implicit resume. Initialization uses zero-QA ROCO-lineage PREtask shared tensors, TRAIN-only vocabulary and positive weight. No calibration/test inference. CPU8, offline dependencies.

Estimated total compute from the measured fixture:4890seconds (~82min), excluding startup, checkpoint I/O and post-run output replay/audits. This is an estimate for these three arms, not all step4 or steps5–7. Do not infer completion from startup.

Historical L/P controls and learning curves have been inventoried with source/checkpoint hashes in historical_LP_evidence.json. Their261TRAIN-image/32validation-image QA curves differ from the new256TRAIN/full321validation cohort; no matched cross-cohort ranking or architectural superiority claim follows. Post-run metric/bootstrap audit is implemented in audit_step4_results_v3_1.py; execute only after all three completion files exist. Fresh-process full quality replay and final B/V/L/P synthesis remain required before step4 exit.

The existing heartbeat was reactivated with explicit30minute progress reporting and updated authorized task context. No new five-seed or screening run is automatically released by these gates.
