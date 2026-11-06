# Lens native-asset trainability gate v1 — prospectively fixed 27 October

Technical acceptance only; no quality release. Use the already fixed first two eligible TRAIN indices [8,5] (IMG0000044 and IMG0000030), not assessment selection. Native SAM views and all historical receipts remain immutable. Source commit 22a940079c8a6b9d94994c3b950ad4b28e7b253c, native SmolVLM snapshot and isolated runtime v2 are bound by hashes.

## Resource and source mapping

Published processor longest edge 2048 was already exercised on actual views (17 tiles/view, peak 10.05 GiB for one visual case). Before any gradient result, fix longest edge **1024** for the batch-two gate; retain native 512 tile size and image splitting. This is an explicit resolution/resource adaptation, not unchanged native training. FP32 CPU8; two simultaneous distinct-image samples, not accumulation-as-negatives. Enable non-reentrant gradient checkpointing to retain gradients through LoRA within a frozen base. Stop at 20 GiB process RSS or 25 minutes; preserve failures, no automatic retry.

Use exact author LensCollator and pool/DCL/InfoNCE implementations, source LoRA r8/alpha16/dropout .05 and seven target module suffixes. Sample two view types with author seed42+batch index. Retain the source second visual forward (no feature reuse). One AdamW update for alignment CE+.1 DCL, then one for QA CE+.1 InfoNCE, temperature .1, lr2e-4/wd.01/betas(.9,.999)/eps1e-8/clip1, no scheduler or accumulation in this disposable gate. This is not the source full phase budget or quality initialization.

## Existing target provenance

Alignment uses existing FracAtlas dataset.csv binary body-region metadata, verified against sealed TRAIN row region strings: IMG0000044 hand=1 and shoulder=1 -> 'hand and shoulder'; IMG0000030 hand=1 -> 'hand'. This is explicitly image-level region alignment adapted to BoneVQA, not a newly invented organ label or mask-specific GT. Reject absent/unknown metadata; never use the author's 'radiology anatomy' fallback. QA uses sealed existing questions and answers. No annotation boxes or polygons are supplied to model or renderer. No assessment records.

## Required acceptance

Bind source, protocol, runtime, original rasters, actual view files, metadata, sealed TRAIN roles and script before updates. Verify actual collator answer masks and prefix/image/padding exclusion. Individually process each semantic view and prove combined tile order and equal tiles within each case before trusting source pooling. Require finite losses and nonzero contrastive feature gradients with batch two; finite nonzero vision/text LoRA gradients, changes across two updates, frozen base digest/gradient invariance. Save compact LoRA state plus full reconstructed-state digest. Fresh process must reconstruct native base and same adapters, strict reload mutable tensors and reproduce eval QA logits digest/loss exactly. Preserve initial, final and per-step digests and all failures. No inference, clinical efficacy, general pooling correctness or full Lens completion claim from this gate. Public original-only generation and bounded two-stage pilot follow separately.
