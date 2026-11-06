# Fixed frozen-feature fracture-head pilot: completed, no promotion

21 October 2026. Evidence: `frozen_head_public_evaluation_v3_1/paired_artifact_audit.json`, fixed seed42 LAST410. All1432 public predictions replayed exactly. Only four fracture-head tensors changed; noncanonical records remained identical. Public evaluation and replay took302.043seconds. Original TRAIN feature extraction took95.770seconds, followed by exact fresh-process feature replay. The internal cached-head fit timer excludes those costs and cannot represent full-model training speed.

| Metric | Original | Head LAST410 |
|---|---:|---:|
| VQA overall accuracy |90.0838%|89.1061%|
| OPEN accuracy |88.6943%|88.6943%|
| Fracture sensitivity |64.9123%|84.2105%|
| Specificity |93.1818%|83.7121%|
| Balanced accuracy |79.0470%|83.9613%|
| AUROC |0.911018|0.916600|
| AUPRC |0.746712|0.762053|
| Brier, lower better |0.093807|0.115666|
| ECE10, lower better |0.068314|0.155093|

The head found11 additional positives while adding25 false positives. Primary BA difference+4.914points has paired known-component95%interval[-0.330,+10.390]points. AUROC interval also includes zero. Secondary AUPRC difference interval[+0.000470,+0.033349] is exploratory, unadjusted for multiple endpoints. Overall VQA difference interval[-1.831,-0.141]points, specificity and probability calibration also worsen. These secondary findings do not establish clinical superiority or justify selecting a new threshold on inspected model-validation.

The fixed pilot is complete with a sensitivity/specificity tradeoff, not a successful overall replacement. Preserve both checkpoints. Keep original native B as provisional reference. Do not expand head epochs/seeds or launch another loss search solely from these results. No calibration/test inference occurred. This single-seed inspected development fold does not provide independent confirmation. Recorded component bootstrap uncertainty does not account for training-seed variability or unknown patient relationships.

Technical qualification of the cache and head-only optimization supports reuse of this acceleration method for explicitly frozen-feature experiments. It is not an efficacy contribution or a substitute for full-model fine-tuning.
