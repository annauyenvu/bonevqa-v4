# Fixed fullfold0 LAST comparison

Both matched seed42 arms finished2epochs/1438updates on11502TRAIN QA, with the same125-answer vocabulary, initial checkpoint, schedules, auxiliary losses and optimizer recipe. Native took4003.35seconds and uniform5296.00seconds including epoch validation and saves. No additional training, epoch selection, threshold fitting, calibration/test inference or external downloads were used.

Fresh-process strict reload reproduced all1432 prediction records and returned probabilities exactly for EACH arm. Recomputed metrics match every trainer metric exactly. Evaluation plus replay/statistics took1070.789seconds (17minutes51seconds). Independent artifact audit status is recorded separately in fullfold0_fixed_LAST_evaluation_v3_1/independent_artifact_audit.json; do not infer that audit passed merely from this narrative.

| Fixed LAST metric | Native token mean | Sequence uniform |
| --- | ---: | ---: |
| QA exact |1290/1432 (90.08%)|1288/1432 (89.94%)|
| OPEN exact |557/628 (88.69%)|553/628 (88.06%)|
| CLOSED exact |733/804 (91.17%)|735/804 (91.42%)|
| Canonical fracture AUROC |0.911018|0.913610|
| Canonical fracture AUPRC |0.746712|0.753048|
| Sensitivity at raw0.5 |37/57 (64.91%)|37/57 (64.91%)|
| Specificity at raw0.5 |246/264 (93.18%)|247/264 (93.56%)|
| Balanced accuracy |79.05%|79.24%|
| Brier (lower better) |0.093807|0.091206|
| ECE10 (lower better) |0.068314|0.059210|

Primary uniform-minus-native OPEN difference is−0.637 percentage points, paired known-component bootstrap95% interval[−1.558,+0.319]points. Overall difference−0.140points, interval[−0.768,+0.417]. AUROC difference+0.002592, interval[−0.001423,+0.006807]; AUPRC difference+0.006336, interval[−0.005339,+0.021128]. All ten reported QA/clinical metric intervals include zero. Exactly500 paired resamples, seed20261021,314known components; one training seed does not quantify training-seed uncertainty. No evidence from this fixed screen establishes uniform superiority or equivalence.

Clinical limitation: both arms miss20/57 positive images at the fixed raw0.5 threshold. Nineteen missed image IDs are shared; one native-only and one uniform-only false negative exist. This is not evidence of particular clinical subtypes or annotation errors. Epoch1 sensitivity was higher, but fixed LAST remains the primary endpoint; do not choose epoch1 or tune a threshold on this inspected development cohort to disguise the tradeoff. QA exact accuracy is not clinical fracture sensitivity.

Earlier bounded pilots used different TRAIN coverage, vocabulary and class weights; their lower point estimates cannot identify which change caused this full-TRAIN result. All321images/1432QA and fold0 are already development-informed, not untouched/patient-independent confirmation. Data, historical failed artifacts and frozen technical sources remain preserved. No Final Solution, module contribution claim, fivefold study, production release or full-step3 completion follows from this pair.

Next safe work is a documented diagnostic of the persistent sensitivity errors using existing labels and TRAIN-only mechanism controls, followed by a prospectively fixed candidate if justified. Do not automatically add epochs/seeds, remove auxiliary losses, tune modelval thresholds, or use annotation ROI/GT routing as a production input. Independent confirmation/calibration and full detector partition/proposal coverage remain separate requirements for any eventual deployment claim.
