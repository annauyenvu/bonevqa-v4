# B retrain on fold 0: analysis plan

Written 2026-11-01 before B seed 43 finished and before any v5 B run was stacked with a detector on test.

Question: how much does the system's test result on fold 0 move when B itself is retrained?

- B runs: sealed B, v5 B seed 42, v5 B seed 43 (the last two trained on MPS, 2 epochs, same fold-0 data).
- Stack for all runs: [B, flag, s1], logistic ridge 1, fit on calibration + model_val, applied to test. The head-refit column is left out because the v5 runs do not have it, so sealed B here is not the locked 0.834.
- Detectors: seed 42 and seed 43, epoch 6.
- Report for each of the 6 pairs: test AP, QA after the consistency layer, contradictions, each with V4 fold 0 alongside. For B seed 43 with detector 42, also AP minus V4 and QA minus V4 with a component bootstrap (2000 draws, seed 20261105).
- No pass/fail threshold. One fold, descriptive only; the range across B runs is the result. No configuration is chosen from it and the locked system does not change.
