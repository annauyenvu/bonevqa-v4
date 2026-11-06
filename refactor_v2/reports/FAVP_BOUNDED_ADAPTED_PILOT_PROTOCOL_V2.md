# FAVP bounded adapted native-asset pilot V2 — prospective resource amendment

V1 design is preserved and was never execution-released. Before any selected validation image generation or answer evaluation, measured public decoding cost was approximately252seconds/20tokens/image. V2 keeps32TRAIN but reduces validation to the first8canonical image IDs so PRE/epoch1/LAST/replay and64updates can fit the unchanged4hour wall budget. This is based only on compute timing, not validation outcomes. Eight examples are an engineering endpoint, not efficacy evidence.


Status: **design frozen, execution NOT released**. Release requires accepted image/question-only generation and actual automatic-proposal caches for the fixed8-example model-validation subset, fresh initialization binding, optimizer gate, dependency/disk checks and public evaluation/replay implementation. Do not silently fill unmet gates.

## Purpose and interpretation

Execute a finite engineering/learning pilot of actual native EVA/QFormer/Vicuna assets with source-sized EVA/LM LoRA, not claim original FAVP reproduction. Source stage-1/2 image-to-Vicuna projection is unavailable; use a fresh random projection and label the experiment de-novo BoneVQA adaptation. Native SAM's complete256-image TRAIN census had zero accepted proposals at unchanged RAD settings, so local-versus-original TRAIN inputs are identical. Do not spend duplicate runs on an uninformative local on/off comparison or claim local-branch benefit. Original-method performance/failure is not established by this pilot.

## Fixed data and budget

- Existing sealed TRAIN and model_val manifests and component lookup under reports/step4_localization_v3_1 remain unchanged. Never load calibration/test or invent labels.
- Only the existing canonical question `Is there a fracture in this X-ray?` is used. This is a disclosed binary-question pilot, not the full source VQA task or historical1,164-record/41-class endpoint.
- Select the first32unique TRAIN image IDs and first8unique model-validation image IDs in lexicographic order from their respective canonical records, independently, **without selecting by answer, proposal count, component outcome or model performance**. Require one canonical record per selected image, disjoint role IDs and complete original-image binding. Freeze selected IDs/indices before any new subset proposal generation or answer evaluation. Do not replace cases with no proposals, unsupported generations or missing outcomes.
- One seed42, batch1, two epochs, exactly64optimizer updates. No search over seeds, budgets, thresholds, generation settings or checkpoint. Shuffle the32TRAIN indices with seed42+epoch for each epoch; save schedules before training.
- Compare fixed PREupdate and LAST2 on the same8model-validation examples; epoch1 is a learning-curve observation only, never checkpoint selection. No extra method/control arms are implied. This is model-development-informed validation, not independent patient/clinical evidence.

## Model and optimizer

- Fresh seed42 initialization: pinned pretrained EVA/QFormer/query tokens/Vicuna; zero-B EVA LoRA39qkv rank4/alpha8/dropout.05; LM LoRA64q/v rank8/alpha16/dropout.05; trainable QFormer/query tokens and fresh768→4096projection. Frozen SAM, EVA base/vision norm, LM base. Do not load disposable gate updates.
- Keep explicit CPU policies from accepted gates: FP32EVA; FP16stored LM base/activations with temporary FP32 frozen-linear compute and FP32 low-rank branches; checkpointed training. Runtime verified EVA attention correction invokes LoRA instead of direct qkv.weight-only F.linear; current Transformers cache_position compatibility and tokenizer semantic BOS/EOS/PAD reconciliation are disclosed. Hash-bind all runtime adapters and pinned sources.
- AdamW initial/base LR3e-5, weight decay.05, betas(.9,.999), eps1e-8; six-step linear warmup from1e-6 to3e-5 then cosine decay to1e-5 across64steps; gradient norm clip1. These are a prospectively fixed short-budget adaptation of source RAD LR/decay settings and200warmup steps, not identical author training. Gate actual AdamW/scheduler, not only SGD, before release.
- TRAIN objective: source-style answer-only causal CE, prefix labels-100, existing normalized yes/no answer plus `###`; lowercased question, source prompt template. No consistency/extra clinical labels or annotation views.

## Public inference and endpoint

- Inference signature receives only original image and question. Automatic SAM is frozen and uses unchanged source RAD thresholds, including zero-proposal original fallback. A hash-bound public-image proposal cache is permitted; caches for all8 selected validation images must be generated prospectively with no target use and preserve all proposals. No annotation ROI or forced local view. Validate nonempty renderer/local order if naturally present; otherwise state unavailable.
- Greedy generation, one beam, at most20newtokens, min_length1, use cache; tokenizer semantic BOS/EOS/PAD explicitly supplied. Decode using source-style split at`###`/Assistant marker and existing normalize_answer. No teacher-forced answers in generation or post-hoc label-dependent answer parsing.
- Primary endpoint: strict normalized free-generation exact accuracy on8canonical validation answers. Report empty, truncated and non-yes/no outputs without removing them; they count incorrect. This protocol does not introduce or tune a clinical AP score. Report question scope and answer-class counts descriptively only after selection is frozen.
- Final versus PRE paired component bootstrap:2,000draws, seed20261103, fixed existing component lookup, exploratory95%interval for accuracy difference; components are not verified patients and one training seed does not estimate seed uncertainty. No promotion decision or clinical claim.

## Completion and resources

Before quality execution: actual optimizer/scheduler gate, source/trainability/frozen-state checks, fresh strict replay; bind common updates0state, data/schedules/code/assets/public caches and generation config in a release. Confirm8GiB disk reserve after projected outputs. Store compact trainable state only; no full7B duplicates or optimizer checkpoints needed for this non-resuming budget. Never delete historical artifacts automatically.

Measure actual updates/generation before ETA; cap the pilot worker at4hours wall time. Failure/limit must be preserved, not trigger automatic expansion. Completion requires64updates, complete8-example PRE/LAST generation, fresh-process exact final replay, independent saved-output recount, component uncertainty, learning curves inspected visually and separate closure seal. Then proceed to Lens, with original-stage/proposal limitations explicitly retained. Broader FAVP training would need a new prospectively justified budget, not an automatic continuation of this pilot.

