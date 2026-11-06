# Reconstructed SLAKE PF: bounded protocol and release status

## Completed sequential integration checks

Receipt: `pf_slake_integrated_v1/receipt.json`. Actual BiomedCLIP/PubMedBERT backbone, original two TRAIN images IMG0000044 and IMG0000030, source-layout 41-class head, six interleaved MM/LF layers followed by reconstructed PF. The node names produce 911 nonpadding PubMedBERT tokens, not the 2002 RoBERTa tokens from the earlier standalone component fixture. Mean raw embeddings exclude padding and special tokens.

The existing cache byte binding and B_PREtask adapter hash were verified. Three inference modes use identical pre-update parameters; digest 203a8b11ba965bf08abc489686d1a21d4ff30c629302bdb8a9248f5092a27d1d. No trained historical pilot is used as initialization.

Two disposable train-mode AdamW updates of the real-graph route completed. PF, node pooling, combined pooling, LF, GM, answer head and visual LoRA had finite nonzero gradients and changed parameters. Answer loss reaches PF; GM consistency loss does not reach PF. Frozen parameters remained unchanged. Full-state fresh-process replay reproduced logits exactly for real, rewired and bypass routes. Public forward accepts images and questions only. Total gate time 16.88 seconds; this is not a quality-run ETA. Post-update checkpoint is technical evidence only and must never initialize a quality run.

## Important unresolved gate

At initialization, maximum answer-logit changes were only 5.96e-7 for rewiring, 1.31e-6 for GAT bypass, and 4.77e-7 for reversing node-name order. Nonzero float32 differences alone do not establish useful graph sensitivity. The script's nonzero assertions are wiring checks, not a passed semantic intervention gate. Quality release remains FALSE.

Next diagnostic must isolate graph features, PF attention outputs and pooled slots, use higher precision/repeated deterministic controls to distinguish numerical noise from structural effects, and inspect query/key attention collapse. Preserve author behavior; do not alter GAT masking or tune against validation outcomes to obtain larger differences. Stop and report if meaningful sensitivity cannot be demonstrated.

## Prospective bounded quality design (inactive)

Three arms: real graph, degree-preserving rewired graph, explicit GAT bypass. Same node labels/order, same 409-to-4 and 36-to-1 pooling, same initialization and TRAIN batches. Real versus rewired is the primary structural comparison; bypass changes active computation and is secondary. Graph409 is reconstructed SLAKE, not author577. This is a shared-backbone port with random new MM/prompt weights, not a native M3AE reproduction.

Use the existing sealed 256-image TRAIN / 321-image model-validation roles (1164/1432 QA), seed42, four epochs, 292 optimizer updates per arm, batch4 with accumulation4, AdamW lr1e-4 weight_decay0.01, clip1, eta0.1. Select LAST epoch4; no thresholds, epochs or seeds searched. Preserve the existing class-scaled BCE and batch-SUM raw target consistency with TRAIN-vocabulary-fixed padding. Closed41-class accuracy and question strata, fixed yes/no margin endpoint threshold0.5; unsupported answers count incorrect. Primary paired AP difference real-minus-rewired, 2000 component bootstrap with fixed seed20261101; components are not verified patients. Report one-training-seed and endpoint limitations.

Before execution: resolve the sensitivity gate; run each control's actual train-mode update/frozen/reload checks; persist common PRE-update initialization; seal complete dependencies, schedule/configs and original-image evaluator; check disk budget and retain full1432-record fresh replay per arm. No quality jobs were launched by this protocol. Historical seals remain unchanged.
