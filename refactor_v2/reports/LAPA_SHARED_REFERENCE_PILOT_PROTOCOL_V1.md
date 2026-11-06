# Prospective shared-backbone reference pilot

Status: design fixed before quality outputs; execution NOT released.

This is a LaPA GM/LF mechanism port on BiomedCLIP/PubMedBERT224 with randomly initialized author cross-modal layers. Native M3AE reproduction and PF are excluded from this release. The existing generative Native B is contextual evidence, not the matched baseline of this classifier experiment.

## Arms and attribution

Five arms: carrier baseline; TRAIN-bank GM/LF with eta0; TRAIN-bank GM/LF with eta0.1; random-bank GM/LF with eta0; random-bank GM/LF with eta0.1. All use the same author-layout 41-class head and cross-modal initialization. Each starts from a byte-identical saved shared initialization and identical visual adapters. No pilot-trained weights initialize another arm.

The carrier arm computes the interleaved prompt route but omits its contribution and consistency loss. This preserves dropout draw order and the shared computation schedule. Unused prompt parameters must receive no gradient and no optimizer update. It is a compute-matched carrier control, not native paper baseline reproduction. Report inactive parameters explicitly.

Random-bank controls retain 319 tokens, width768 and per-token norms of the TRAIN raw-word bank. Generate fixed Gaussian directions with seed20261025; no clinical interpretation. Only bank values change; targets for consistency remain the true TRAIN answer word embeddings in all eta0.1 arms. Verify shapes/norms and initialization before release. Real-minus-random within each eta isolates bank information more closely than real-minus-carrier alone.

## Data and objective

Reuse the sealed step4 cohort: 256 TRAIN images/1164 QA and 321 model-validation images/1432 QA. The 41-class vocabulary is TRAIN-only. No calibration/outer assessment access in this pilot. Original images and questions are the only per-record public inputs.

Mean BCE multiplied by41, plus eta times batch-SUM raw-word cosine distance. Tokenize each target with per-answer special tokens and pad to the maximum TRAIN-vocabulary token length17; include padding in the source-style unmasked mean. TRAIN bank concatenates per-answer tokens including special tokens. Both are disclosed preprocessing adaptations. Unknown validation answers count incorrect and are separately reported, never added to the vocabulary.

## Bounded optimization

Seed42, four epochs, LAST4 only, 292 optimizer updates per arm. Reuse and verify existing component-aware microbatch schedules: batch4, accumulation4, with the same final partial accumulation rule. AdamW lr0.0001, weight_decay0.01, max_grad_norm1, float32 CPU8; no scheduler or automatic early stopping. Divide each microbatch objective by the actual number of microbatches in its accumulation window; preserve the source batch-SUM consistency within each microbatch. This defines the port's effective loss scaling and must be reported; it is not equivalent to computing the author's SUM on one batch16.

Frozen backbone stays in eval mode; trainable cross-modal/prompt/head modules use train mode with source dropout. Visual LoRA receives gradients through the frozen backbone. Freeze text embeddings, PF and unused graph reductions. Freeze carrier prompt parameters. Do not infer frozen status merely from the optimizer list: check parameter hashes and gradients.

## Evaluation and decisions

Report closed-vocabulary exact accuracy overall/by question family, unsupported-answer coverage, fracture AP/AUROC and sensitivity/specificity/BA. For binary fracture questions use the yes-minus-no logit margin, sigmoid probability and fixed0.5 decision. Report other-class argmax separately; do not silently equate it with binary prediction. Verify canonical yes/no vocabulary IDs before running. No threshold calibration/search.

Use paired component bootstrap2000 draws, seed20261031, for LAST4 AP differences and accuracy differences. Report exploratory intervals with multiplicity disclosure. No automatic promotion, additional epochs or seeds; assess information contribution from matched real/random contrasts and carrier comparison without selecting the best epoch.

## Execution release requirements

Implement all five routes, materialize matched initialization/bank/schedules, verify data and source hashes, and run train-mode two-update gates per arm with actual optimizer/accumulation. Verify isolated answer/consistency routing, finite updates, frozen invariance, carrier inactive parameters, original-input-only inference and full fresh-process output replay. The existing eval-mode SGD two-record integrated gate is supporting evidence but does not certify this training configuration.

Seal configs, initialization, complete dependencies and gates before quality execution. Measure actual train-mode update/evaluation costs before estimating runtime. Preserve failed runs and all historical closures. Report compute differences and development-informed data limitations; no native-reproduction, independent clinical or production claim.
