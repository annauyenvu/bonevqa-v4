# Step 3 exit review after grouped outer evaluation

22 October 2026, ICT. This supersedes the outstanding grouped-experiment dependency in `STEP3_EXIT_REVIEW_V3_3.md`. Historical documents and failed artifacts remain preserved. It does not reconstruct or renumber the original seven-step plan from partial records.

## Completed work and decisions

| Requirement from prior reviews | Verified current outcome |
| --- | --- |
| Minimal grouped question and run matrix | Frozen amendment: estimate native B performance, seed42, two-epoch LAST; no architecture superiority claim or inner search |
| Fold-specific initialization/config/schedule and runtime gates | Passed; historical fold0 compatibility/reuse disclosed, four new folds completed |
| Checkpoint/model-validation artifact and replay gates | Passed for all five fixed checkpoints within their documented scopes |
| Separate calibration chronology and executable specification | Fixed estimator/threshold/failure policy before outer execution; calibration-only fitting verified |
| Once-per-image outer evaluation with replay | Completed:4013images,3920knowncomponents,17998QA; all five independent fit/prediction/metric audits passed |
| Frozen uncertainty and independent audit |2000paired within-fold component draws, seed20261021; fresh-process full replay; independent reconstruction of all primary draws and all percentile/count summaries passed |
| Evidence-based quality decision | Raw BA75.26%, sensitivity55.94%; calibration reduces BA1.74pp, paired95%interval−2.65to−0.91pp. Calibration/overallQA improvements do not justify promotion |
| Scientific reporting | `GROUPED_OUTER_RESULTS_V3_1.md` reports results, controls, seed, module scope, timings, limits and negative findings |

The declared grouped experiment is complete. There are zero remaining training epochs or inference/replay jobs in this experiment. Technical completion and a positive scientific result are separate: the current model's detection sensitivity remains limited, and the frozen calibrated variant worsens the primary endpoint. Waiting longer will not change these completed results.

## Scope that remains unresolved

The original seven-step request is not fully reproduced in the local authoritative records inspected here. This review therefore does not certify the whole original step3 or seven-step deliverable as completed. The concrete grouped confirmation/calibration dependency listed in the previous review is now resolved with a development-informed estimation result, not a superiority or untouched independent clinical confirmation claim.

Current T/V/L/P matched screens have not established benefit; this five-fold experiment deliberately uses native B with those modules off. Full-fold V/L detector/proposal coverage is not required for this B-only experiment and has not been silently certified. GM+LF remains the complete LaPA scope; PF is excluded. There is no evidence-backed reason to add large V/L/P training solely to seek a positive result.

Multi-seed robustness, verified patient grouping, unknown base-model pretraining overlap and untouched independent clinical validation remain unestablished. These are limitations, not implicit authorization for new experiments. Historical PMC contamination remains disqualifying as independent evidence; clean artifact lineage alone is not pretraining efficacy.

## Next decision boundary

The completed reports support documenting the fixed native baseline and the negative calibration comparison. They do not establish a clinically ready Final Solution or production input contract. No new architecture, epochs, seeds, loss weights, threshold tuning, detector expansion or production release follows automatically from this review.

A further quality-improvement study needs a concrete, separately frozen mechanism and design, with explicit disclosure that all existing outer results are now inspected. Alternatively, the existing evidence can be consolidated into the paper/report with negative findings and a limited claim. Direction for materially new work must come from the user's intended next step; routine monitoring cannot create that authority.

Use `CURRENT_GATE_LEDGER_V3_69.json` and later closure/status records for current facts. Earlier `module_gate_status_v3.json`, implementation summaries and paper Final Solution documents are historical and do not override the current unresolved quality/production status. The190-artifact original technical seal was verified unchanged during this closure review; its generic `long_training_authorized=false` is not a retroactive failure of the separately scoped completed releases.
