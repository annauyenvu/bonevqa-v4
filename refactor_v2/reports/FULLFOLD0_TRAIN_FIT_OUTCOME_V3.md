# Frozen TRAIN-fit diagnosis: verified outcome

Both fixed LAST1438 models were evaluated on every2568canonical TRAIN image (458positive/2110negative), original image/raw question only, batch4, raw threshold0.5. There were no optimizer updates, threshold searches, new modelval inference, calibration/test inference or label changes. All2568prediction records/probabilities/metrics per arm replayed exactly in a fresh process. Separate artifact audit independently recomputed identities, metrics, confusion counts, AUROC/AP/Brier/ECE and physical checkpoint hashes.

| Endpoint | Native TRAIN | Uniform TRAIN | Native saved modelval | Uniform saved modelval |
| --- | ---: | ---: | ---: | ---: |
| Sensitivity |314/458 (68.56%)|308/458 (67.25%)|37/57 (64.91%)|37/57 (64.91%)|
| False negatives |144|150|20|20|
| Specificity |2025/2110 (95.97%)|2030/2110 (96.21%)|246/264 (93.18%)|247/264 (93.56%)|
| Balanced accuracy |82.27%|81.73%|79.05%|79.24%|
| AUROC |0.916378|0.916874|0.911018|0.913610|
| AUPRC |0.809813|0.810585|0.746712|0.753048|

TRAIN false negatives overlap on140image IDs; native has4additional and uniform10additional missed images. These are observed training-label errors, not independent efficacy evidence. Because sensitivity is already low on TRAIN, the problem is not exclusively an unseen-validation performance drop. This does not identify whether representation, optimization, loss interaction, label difficulty or threshold behavior is responsible; it does not prove that more epochs would solve it. Different cohort sizes/composition make the TRAIN/modelval gap descriptive, not a controlled causal estimate.

Actual firstpass times were163.053seconds native and111.830seconds uniform; the full diagnostic including two-arm fresh-process replay took478.677seconds (7minutes59seconds). Separate artifact audit took1.447seconds,0models/updates. Mixed OPEN-QA validation throughput was not used as a claimed measurement of this all-canonical workload.

Source inspection (not a gradient experiment) confirms the current dedicated fracture head is supervised only on existing canonical yes/no questions using BCE with TRAIN-derived positive weight2110/458; inference routes the canonical question to this head's sigmoid at raw0.5. The saved125-way closed head also receives closed-answer CE. That shared-loss structure motivates measuring the actual TRAIN fracture-head/shared-feature gradients and their interaction before proposing a loss/architecture change. Source inspection alone does not establish gradient conflict, a broken routing path, or a fix.

Report SHA d42aaec9314797bd0433f12c7efdcbf95cb014141ab775fa59b69ee234443fb6. Protocol SHA cdde7531a48fea84470f3ac33a21cc785cf18e8c2687aedfea8ceb9fef7afcc3. Independent audit: fullfold0_TRAIN_canonical_diagnostic_v3_1/independent_artifact_audit.json. Original progress/report pending-audit flags remain historical; the separately bound passed audit closes this diagnostic. No new quality training or full-step3/production release follows.
