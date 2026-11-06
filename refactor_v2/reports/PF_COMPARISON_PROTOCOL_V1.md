# Prospective four-arm reconstructed PF comparison

Scope: shared BiomedCLIP/PubMedBERT port with reconstructed SLAKE409 and randomly initialized MM/prompt blocks. Not native pretrained LaPA. Freeze this design before new quality outputs.

Arms: author_real; masked_real (absent edges blocked, neighbor normalization); residual_real (same masked GAT plus unit graph-query residual); residual_rewired (same residual mechanism, degree-preserving edge control). Parameters and initialization are identical. All retain 36-slot pooling, TRAIN bank and eta0.1. Rewiring does not invent medical facts; it is a control.

Primary contrast: residual_real minus residual_rewired AP. Secondary: residual_real minus masked_real isolates the residual at fixed GAT; masked_real minus author_real evaluates the combined masking/normalization change. These are exploratory, unadjusted contrasts; do not select a winner by validation to expand training.

Same existing sealed 256-image TRAIN/321-image model-validation roles, 1164/1432 QA, seed42, four epochs292updates, batch4 accumulation4, AdamW1e-4 wd0.01, clip1, LAST4. Class-scaled BCE plus batch-SUM consistency eta0.1. TRAIN-fixed target padding; public inference original images/questions only. Fixed yes/no margin sigmoid threshold0.5, closed41-class accuracy and question strata; unsupported answers incorrect. Paired2000 component bootstrap seed20261101. Components are not verified patients; one TRAIN seed does not estimate training-seed uncertainty.

Two actual scheduled accumulated updates and fresh-process checkpoint reconstruction/replay are required for every arm, with all active module changes and frozen-state invariance. Compact checkpoints reconstruct fixed parameters from hash-bound offline sources, validate full-state digests and strict-load keys. Disposable post-update gate states must never initialize quality runs.

Quality execution remains inactive until PRE-update initialization, complete source/data dependency seal, configuration/schedule files, evaluator checks and disk feasibility are complete. Reserve at least8GiB during technical checks. No automatic deletion of historical artifacts, new downloads, installs, extra seeds or threshold searches.
