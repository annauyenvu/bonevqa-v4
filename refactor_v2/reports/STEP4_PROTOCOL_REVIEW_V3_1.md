# Step 4 protocol review — 22 October 2026

Decision: READY FOR IMPLEMENTATION of a bounded development pilot; NOT YET RELEASED FOR TRAINING. Review requested by user after discussion of B/V/L/P localization controls. This is a prospective design recommendation, not evidence of executed experiments. Periodic automation remains paused.

## Concrete findings

The 23 candidate files are recipes, including separate alignment/VQA phases and controls, not 23 completed architecture comparisons. The inspected v.json still points to revised_pmcvqa_alignment_v3/best.pt. That historical candidate dependency cannot automatically be reused: PMC quality screening failed and the reference experiments used a different, audited initialization lineage. Select and hash a compatible PREtask initialization derived from the successful ROCO lineage, with zero FracAtlas QA updates; verify common shared tensors and matched newly initialized heads across arms.

V_full321_proposals_v3_1 is explicitly evaluation-only: 321 images, 124 nonempty, 197 empty, training_released=false. It is not a training proposal release. Detector partition and preprocessing must be bound to the selected cohort. A detector trained on outer-fold TRAIN may be used for that fold's TRAIN proposals, with in-sample detector exposure disclosed; it must never have trained on model-validation/calibration images. Out-of-fold detector proposals would answer a different training-distribution question and must not be silently substituted.

Existing fold0 model-validation is inspected development data. Existing outer data has also informed development. No subset can be renamed an untouched confirmation holdout. This pilot supports development decisions only. Existing five-fold results and closure remain historical fixed evidence.

## Stage A: diagnostic implementation, no training required

Use existing saved predictions and labels to report hardware × fracture_count (one versus multiple; retain exact counts in supplementary output), with TP/FN and TN/FP where supported. Report missing/empty cells and component bootstrap intervals; region/view adjustment is descriptive with explicit common-support restrictions. No causal shortcut claim from association alone.

For the existing 321-image proposal cohort, report positive-image proposal availability, GT-box recall at IoU 0.25 and 0.5, number/area of proposals, and coverage of baseline FN. Existing GT localization is audit metadata only. Verify coordinate transforms and image hashes before overlap calculations. Missing annotations are unknown, not negatives. Evaluate all images; proposal-present and hardware subgroups are secondary, not an eligibility filter. Existing 197-empty statistic alone does not establish poor positive-image recall.

## Stage B: bounded B/V/placebo development pilot

Use fold0 TRAIN and model-validation only. Construct a deterministic component subset from TRAIN, seed20261022, targeting 256 images by whole-component selection (report actual count), stratified by existing fracture label where feasible; preserve all QA rows for selected images. Select without model errors, proposal success, or outer outcomes. Keep full321 model-validation. Freeze exact IDs, source hashes, QA schedule and class weights before fitting. No calibration/test loader in training configs.

Arms: native B; V-real; V-location-placebo. Same shared initialization, TRAIN rows/order, optimizer, learning-rate schedule, losses and fixed four-epoch horizon. Batch4, accumulation4, AdamW lr1e-4, weight_decay0.01, clip1, CPU verified reference, seed42. Record integer update count after manifest construction, before execution. Save epoch0/1/2/3/4 curves; fixed LAST4 is primary comparison. No extending the run after inspecting its curves. Report wall time, parameter counts, image/view/token exposures separately; B has different view cost and is an incremental-system comparison. V-real versus placebo is the primary location comparison.

Real/placebo proposals have identical availability, counts, sizes and processing budget. Frozen geometric displacement must be valid in each image; record incidental GT overlap after generation without regenerating until the placebo misses GT. Keep all empty-proposal images in both arms. Input at inference is original image/question plus automatic proposals under the declared V contract; no annotation regions/answers/boxes as input.

Primary endpoint: model-validation canonical fracture AP, real minus placebo, paired known-component bootstrap2000 seed20261022. Secondary AUROC, sensitivity/specificity/BA at raw0.5, Brier/ECE, VQA CLOSED/OPEN, subgroup counts and curves. Report full cohort, including images with no proposals. No significance claim for the many exploratory subgroup comparisons.

Sensitivity at equal specificity may be reported as a descriptive model-validation ROC comparison at prespecified specificity0.90; it is not a deployable threshold or independent estimate. No threshold learned from this pilot is deployed. A later operating-point study must separately freeze calibration/assessment roles and acknowledge historical exposure.

Gradient/update/routing gate: demonstrate real local residual reaches fracture logits, gradients and updates are finite/nonzero where expected, zero/empty residual fallback is consistent, and labels/annotation metadata cannot change public prediction when images/questions/proposals are held fixed. Include changed-proposal inference interventions with frozen weights; distinguish intervention sensitivity from beneficial learning. Preserve failures.

Expansion decision (development budget rule, not a superiority test): if real-minus-placebo AP point estimate is positive, run paired seeds43/44 under the identical frozen recipe. If nonpositive, stop automatic expansion and diagnose proposal coverage/curves; no automatic epoch or loss search. Report all seeds, including failures and negative outcomes. A positive point estimate only authorizes replication, not module promotion. New fold validation remains separate.

## Conditional later stages

L: real/original-teacher/location-placebo arms each receive the same fixed two alignment plus two VQA epochs. Each VQA arm starts from its own alignment checkpoint. Matched initialization and exposures within phase; original-image inference. Compare L-real against its same-phase controls first; one-phase B is descriptive. Freeze this release separately after Stage A/B review, not automatically as a batch of 23 jobs.

VP: establish a matched B/V/P/VP factorial with within-architecture P-unconditioned and V-placebo controls. Test interaction delta=(VP−V)−(P−B) on the same endpoint/cohort/seeds, with paired component uncertainty. Shared state must match where architecture permits; report added capacity. One combined arm exceeding B alone cannot establish synergy. P scope GM+LF, no PF.

V+L remains a new hypothesis requiring its own design. T is deferred pending a specific mechanism beyond generic template addition. No automatic progression based solely on a favorable subgroup.

## Remaining execution checks

1. Materialize cohort/configs and assert disjoint known components and exact-byte hashes; freeze source/initialization/QA-order hashes and integer update budget.
2. Bind correct PREtask initialization, excluding historically failed PMC candidate initialization; strict shared-state comparison and allowed-head initialization audit.
3. Export automatic proposals for the actual TRAIN subset and full model-validation with detector provenance, geometry parity and empty handling verified. Do not relabel evaluation-only export as training-ready.
4. Run tiny disposable routing/update/reload/public-input gates and a measured throughput fixture. Only then estimate elapsed time and release the bounded run.
5. Record every artifact in a new execution release; preserve previous frozen source and checkpoints. Implement wrappers/new files rather than modifying prior sealed experiment code.

The review removes no data-quality safeguards and does not require a positive sensitivity result as a prerequisite for a development pilot. It identifies concrete preparation work; scientific effectiveness remains an outcome to measure.
