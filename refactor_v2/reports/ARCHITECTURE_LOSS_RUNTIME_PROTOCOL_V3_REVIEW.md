# Revised v3 CPU technical reference — scientific launch NOT sealed

This document records the verified CPU implementation. It is included in STEP3_TECHNICAL_REFERENCE_SEAL_V3.json, a technical snapshot, NOT a long scientific-run release or prompt-efficacy claim. Historical configs/results remain preserved. The23 resolved candidate recipes have passed structural data/loss/lineage audit. Fresh upstream checkpoints, complete new-fold detector proposals and individual run release checks remain required before corresponding long experiments.

## Data and initialization

FracAtlas uses the five component-grouped development folds, each with separate TRAIN, model-validation, calibration and evaluation manifests. Evaluation has already informed development; no untouched-test or expert-clinical validation claim. Existing GT only. PMC train_lon and historical PMC checkpoints are excluded from independent-evidence claims. Offline base weights, caches, common initialization and revised pretraining stages have distinct provenance roles: a fresh common-initialization smoke is NOT a completed ROCO/PMC alignment checkpoint.

Planned upstream order remains sealed ROCO caption alignment, sealed PMC generative alignment, then fold-specific VQA. Each stage must use its own validated data adapter. FracAtlas component batching must NOT leak into ROCO/PMC loaders by YAML inheritance. Base-model pretraining independence remains unknown.

## Architecture and production inputs

B: original EXIF-canonical image plus raw question; V/L/P/T off. Shared BiomedCLIP visual and PubMedBERT encoders, visual LoRA, QFormer/prefix-to-Qwen bridge, multimodal fusion, closed-answer classifier and separate canonical fracture head. The user provides no region labels or GT answer at inference.

V: detector-derived local views, QFormer paired joint-minus-global delta, single declared fusion-only route; no annotation ROI and no dual injection. Detector must be trained on the current outer-fold TRAIN partition with matching preprocessing/proposal provenance. Empty proposals yield zero local residual.

L: training-only Localization Lens alignment to the shared visual representation; automatic inference uses the original image. It is not a direct inference lens injector. New candidate recipes use real/original-only/matched/shuffled arms, each with two alignment epochs followed by two VQA epochs, identical sampler, losses and horizon within phase. Each VQA arm initializes from its OWN corresponding alignment checkpoint. The primary semantic comparison is L-real vs these controls; L-vs-one-phase B is secondary and NOT an equal-compute claim. Same exposure/step horizon does not guarantee equal measured FLOPs. Historical separate Lens initialization must not be silently reused.

P: GM+LF only, no PF graph. Train-only vocabulary/answer-bank state; gated residual on the answer-consumed fused representation. Answer GT embeddings enter the consistency loss during training only. Bank/prior must round-trip without overwriting learned state. Random-bank, unconditioned and component-shuffled controls share architecture; bank-off and gate-zero are interventions, not automatically compute-matched models.

T is optional exploratory scope, not a default module: raw-question baseline, exact joint-token-matched irrelevant template and paraphrase controls. No region/anatomy annotation added as hidden metadata.

## Implemented losses (must be frozen explicitly in final resolved recipes)

`L_answer = L_closed_CE + w_lm L_open_token_CE + w_fracture L_binary_fracture_BCE`.

`L_total = L_answer + eta L_GM_consistency + lam L_image_text_contrastive + w_lens L_lens_alignment + w_type L_answer_type`.

Fracture endpoint is the exact canonical image-level presence question, not all questions containing the word fracture. Positive weight derives only from selected TRAIN endpoint rows. Stage1 caption mode uses full captions with explicit caption-CE weight; other VQA stages use open token CE. Contrastive multiple positives are image-grouped; single-distinct-image Lens batches use the audited cosine fallback. ROI-empty/no-supervision batches are counted explicitly, not assigned invented labels. A future clinical performance claim cannot be inferred from loss decrease alone.

## Runtime and matched controls

CPU/no-checkpoint execution is the verified numerical reference; installed torch2.2.2 MPS stochastic checkpoint gradients and resume remain uncertified. Do not label an untested MPS recipe certified because CPU passed. All execution remains pinned and offline; no CLIP fallback, downloads or visual mismatch whitelist.

Common component-balanced QA schedules preserve every row and use the same seed/epoch/order for all matched arms. LF donor contexts and masks follow a bijective permutation to different sealed components. Impossible batches reject; do not selectively drop placebo rows. Parameter count, image/view/token count, optimizer steps, exposures and measured runtime must be reported separately: equal steps are NOT equal FLOPs. Generic/matched_all ROI arms change presence/view budgets and are diagnostic unless separately matched. L's extra alignment phase requires explicit same-phase controls.

Only scoped <=16-row/<=2-epoch diagnostic shuffled-P execution is currently enabled. Long shuffled-P training remains blocked. New candidate scientific rule: shuffled context corruption is TRAINING-ONLY; validation/calibration/evaluation use each image's OWN conditioned context. Thus this compares a corrupted-context training intervention, not a batch-dependent deployment system. Unconditioned LF remains unconditioned in both training and evaluation. The rule is explicit in saved model config, not a post-test threshold choice. The TRAIN-only frozen-model audit passes singleton/batch/reordered checks (max closed-logit difference2.69e-7; numeric tolerance1e-5); this is not clinical robustness or efficacy certification.

## Model selection, calibration and reporting

Use fixed full model-validation, threshold-free AUROC/AUPRC; never select on BA@0.5 over300 QA. Dedicated calibration cohort fits Platt and selects the predeclared >=90% sensitivity operating rule, then freezes both for evaluation. The sensitivity target applies to calibration, not a guaranteed evaluation sensitivity. Keep raw/default and calibrated metrics. Report AUROC, AUPRC, sensitivity, specificity, BA, Brier/ECE and image-clustered bootstrap intervals; matched-seed uncertainty only when actually trained. Negative Platt slope is a warning, not an improved-model claim.

Calibration verifies full sealed QA/GT coverage and disjoint components; runtime binds artifact to checkpoint/cohort hashes. Annotation ROI is permitted only for explicitly named audit pixel interventions, assisted settings or upper bounds, never automatic inference requirements.

## Closure checklist

- Passed technical evidence is recorded in module_gate_status_v3.json; no module superiority inferred.
- New23 fully resolved JSON candidate recipes now bind revised data/common schedule and fresh upstream lineage; historical recipes remain unchanged. Inspect/audit these resolved values rather than names alone.
- Two-phase L/common-budget declaration and shuffled-P scientific evaluation rule are explicit and23/23 candidates pass structural loss/data/init/proposal audit; missing generated inputs are reported, not fabricated.
- CPU source/config regression reference is cryptographically snapshotted. Long-run execution releases remain separate. Missing upstream checkpoints and full-fold detector proposals are execution-stage prerequisites, not permission to substitute historical artifacts.
- Only afterward launch bounded screening; only clinically/causally passing modules receive matched seeds or enter the Final Solution. Final module selection remains OPEN.
