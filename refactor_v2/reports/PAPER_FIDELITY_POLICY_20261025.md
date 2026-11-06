# Paper fidelity priority — user direction, 25 October 2026

The user explicitly prioritizes following original LaPA, FAVP and Localization Lens as closely as feasible, to support interpretable comparisons of prompt mechanisms. This direction supersedes the earlier preference to prioritize novel input-conditioned P immediately after the bank/consistency diagnostic. It does not authorize unbounded training, new data leakage, replacement of sealed runs, or unverified reproduction claims.

## Preserve current evidence and execution

At inspection, L coordinator 31862/report 31863 remain live, with L_off quality worker 34223 active. P coordinator 32289/report 32290 are waiting for L analysis closure. Process access required approved external inspection after sandbox ps denial. Do not duplicate or alter these locked runs. Finish the already authorized P bank/consistency diagnostic under its existing specification; eta0.01 is historical/diagnostic, not a faithful-paper hyperparameter claim.

Completed V local-token/auxiliary and current L patch-head spatial supervision are BoneVQA extensions. Their negative or positive outcomes do not establish that original FAVP or Localization Lens fails or succeeds. Historical GM+LF evidence does not evaluate full LaPA PF.

## New implementation priority

After the frozen diagnostic, prioritize paper-aligned reference implementations and their technical releases before novel P context adapters, additional auxiliary heads or combinations. Keep input-conditioned P components as unreleased extensions. A reference is not released for training merely because this policy exists: each implementation needs a source-to-code fidelity audit, pinned configuration, provenance, empirical gates and a bounded execution release first.

For each method prepare an explicit mapping of paper equations, author code and local implementation, covering preprocessing, tensor routes/shapes, initialization, trainable parameters, losses/weights, phases, inference inputs and checkpoint selection. Pin author-code revision and weight hashes when available. Resolve paper/code discrepancies explicitly rather than silently choosing one. Missing assets remain documented blockers to the relevant fidelity claim, not permission for undocumented substitution.

### LaPA

Reference the paper https://arxiv.org/html/2404.13039v1 and author repository https://github.com/GaryGuTC/LaPA_model . Preserve GM, target-answer consistency, LF and actual graph-based PF as distinct, auditable mechanisms. Compare B, GM+LF consistency-off/on, and GM+LF+PF with matched controls. Existing TRAIN-only bank and separation of evaluation roles remain mandatory. Verify PF query/key/value direction, node embeddings, graph preprocessing, GAT and concatenation against paper/code; raw SLAKE source availability is not verification of the authors' processed graph. Reconcile paper eta0.1 reported optimum with the existing diagnostic eta0.01 in a prospectively locked reference config, not by changing a running study or searching assessment results. Direct image/question conditioning of GM is a separate later extension.

### FAVP

Reference https://github.com/OpenMICG/FAVP and https://doi.org/10.1609/aaai.v39i9.33047 . Audit automatic SAM-Med2D masks, mask-to-prompt rendering, shared image encoder, local/global token combination, Q-Former/projection and language-model input against the released method. Existing detector-box proposals and residual cross-attention/local MIL auxiliary loss must be labeled substitutions/extensions, not original FAVP. Check locally available weights, licensing, offline feasibility and proposal provenance before releasing any faithful arm. Annotation-derived test ROI is not an acceptable substitute for automatic proposals.

### Localization Lens

Reference https://github.com/CVLABLUMS/localizationlens and https://doi.org/10.1007/978-3-032-05114-1_33 . Audit original/single-color/multicolor/masked representations, view sampling, attention fusion, token reduction, DCL/InfoNCE phase schedule and original-only inference against paper/code. Do not substitute simultaneous patch-head BCE spatial supervision for the original lens/alignment mechanism. Audit whether existing box annotations can legally implement each required representation; box-derived lenses are weak-supervision adaptations, not segmentation-mask reproduction. Never invent segmentation or clinical ground truth.

## Comparison and claim boundaries

Maintain two explicitly separate comparisons: (1) method-faithful references, with each method's own baseline and disclosed native assets/settings where feasible; (2) mechanism-faithful ports to a shared BoneVQA backbone, cohort and evaluation. Native methods use different backbones/pretraining, so their raw scores do not isolate prompt effectiveness. Shared-backbone ports improve attribution but are not exact original reproductions. Report both fidelity and effective training/compute budgets; equal updates alone do not ensure equal compute or convergence.

Retain matched semantic/placebo/capacity controls as additional scientific checks without rewriting the reference mechanism. Freeze thresholds/checkpoint rules and calibration roles before assessment. No automatic seed/epoch/threshold expansion to chase gains. Continue factual 30-minute updates under the existing monitoring request. No claim about original-method superiority, independent clinical confirmation or production readiness without the required verified evidence.
