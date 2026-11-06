# Five-seed repeated development evaluation — locked23October2026

User authorized steps5–6. Freeze this design before the new P/L screening outcomes or any step6 reserved-role inference. This is development-informed repeated evaluation, explicitly not untouched independent clinical confirmation. Full seven-step/production release is not implied.

## Candidate selection and seeds

Step5 uses the common256TRAIN-image1164QA cohort and full321-image1432QA model-validation. Audited B/V seed42 LAST4 pilot outputs are reused with prior inspection disclosed. New P-real/random/unconditioned and L-real/original/placebo complete the common-cohort matrix. Each one-phase arm has292updates; each L arm has146alignment+146VQAupdates. Native B is not an equal-phase L control.

Candidate AP must exceed B and each own matched control by at least0.005 at fixed LAST. Select at most one, highest AP with alphabetical tie-break. This is a pragmatic budget rule, not a significance claim. Report all rejected arms. If none passes, confirm baseline stability only and record no module promoted. Do not add VP/T or new loss/threshold searches automatically.

Step6 seeds are42,43,44,45,46 for baseline and the selected candidate plus its matched controls. Reuse compatible audited seed42 LAST rather than retrain it. For seeds43–46, instantiate a new seed-specific PREtask model and regenerate task heads/module initialization from the seed, load the same audited ROCO upstream non-task tensors, and hash shared initial states between each baseline/candidate/control pair. Preserve TRAIN-only vocabulary and positive weight. Do not clone seed42 task-head randomness as five independent initializations. Use per-seed common component schedules preserving all rows; fixed4epochs/292updates for one-phase arms. L keeps its two-plus-two phase structure and within-phase matched controls.

## Calibration and assessment

Use the existing fold0 reserved calibration321images and outer803images, with exact source hashes bound before inference and known-component disjointness checked against this TRAIN subset/model-validation. Both roles have historical development exposure; label all results accordingly. No new cohort acquisition is assumed. Model selection uses model-validation only and must finish before new calibration fitting.

For every frozen seed/arm checkpoint, fit Platt on calibration canonical rows only: LogisticRegression(C=1,solver=lbfgs,max_iter=1000,tol=1e-8,random_state=42,fit_intercept=True,class_weight=None) on logit(clip(p,1e-6,1−1e-6)). Keep raw and calibrated threshold0.5. Single-class/nonfinite/convergence failure is preserved, no hyperparameter repair from assessment labels. A nonpositive slope is reported as a failed monotonic-calibration gate; retain raw without a replacement fit. No sensitivity-target or threshold search.

Evaluate public original images/questions for B/P/L, automatic real/placebo proposals for V; L boxes only in TRAIN alignment. Bind proposal provenance/coverage before any V assessment, never annotationROI. Existing GT only, LaPA GM+LF noPF. Calibration transformation changes canonical fracture records only.

Report per-seed AP,AUROC,sensitivity,specificity,BA,Brier/ECE and VQA; raw and calibrated. Report five-seed mean/sampleSD/range and each paired candidate-minus-baseline/control difference. Do not stack the same803images across seeds as4015independent images. Use paired known-component bootstrap2000draws seed20261023, identical sampled components across seeds/arms, and average metric differences across fixed trained seeds per draw. Describe this interval as conditional on those five fitted seeds, not all training randomness or verified patients. Retain all undefined counts, negative outcomes and exact fresh-process replay checks.

## Execution gate

Before launching new seed training: materialize/hash seed-specific configs and initialization, verify shared tensors and real bank/control differences, run disposable loss/gradient/update/routing and reload tests, freeze integer update caps and source/data dependencies. New seed fits and reserved inference require separate execution releases. Do not call this document an executed confirmation. Record actual timings before ETA updates. Preserve step4 and prior grouped seals unchanged.
