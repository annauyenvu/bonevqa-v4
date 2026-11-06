# Fixed fullfold0 B/native–B/uniform screen and real accumulation timing

2026-10-21 ICT. The next design is now concrete: seed42, two arms, full11,502-QA TRAIN/2,568images, same125-answer vocabulary, same zero-QA initial checkpoint, same batch4/accumulation4/LR/loss auxiliaries and complete component-balanced schedules. Exactly2epochs,719 optimizer updates/epoch, maximum1,438updates/arm. No additional module or rarity weighting. This is a bounded fold0 development screen, **not** the five-fold nested confirmation study.

Primary fixed contrast: uniform minus native OPEN exact accuracy over all628 OPEN questions in the1,432-QA model-validation cohort, using LAST epoch2/update1438. No best epoch/seed/threshold/template selection. Report overall/CLOSED accuracy and canonical321-image AUROC/AUPRC,sensitivity,specificity,BA,Brier,ECE at raw.5 as secondary.500paired known-component bootstrap resamples,seed20261021; no multi-seed uncertainty claim from one training seed. Fresh-process public prediction/probability replay remains mandatory. No calibration/test model inference or target-based weight/vocabulary fitting.

Full1432 and earlier142 already informed development. This prospectively fixes the new full-TRAIN pair, not an independent discovery or retroactive replacement of the original142 primary. An OPEN gain alone is not clinical benefit, production readiness or permission to add seeds/epochs. The full TRAIN vocabulary/class weights differ from the old1173-QA pilot, so the causal contrast is within the new matched pair, not an attribution from old-versus-new point accuracy.

## Measured TRAIN accumulation timing

First16microbatches of the fixed component schedule were used for each arm,64existing TRAIN QA/32OPEN, with accumulation4,AdamW,warmup/cosine horizon1438,clip1. Four disposable updates/arm;8total, all resulting weights discarded,0checkpoint saves. Both arms start from identical263 tensors. All tracked QFormer/prefix/visualLoRA/LLMLoRA accumulated gradients and actual updates were finite/nonzero; all trainable states finite. No reserved-role model inference.

| Arm | TRAIN loop for64QA | Mean seconds / accumulated update | Linear TRAIN-only projection for1438updates |
|---|---:|---:|---:|
| Native token mean | 13.496sec | 3.374 | 80.86min |
| Sequence uniform | 13.143sec | 3.286 | 78.75min |

Total benchmark35.541sec including loads/checks. The loop measures actual input decoding,forward/backward,accumulation,clipping,optimizer and scheduler rather than extra diagnostic autograd passes. It is still only the first64rows: token lengths, thermals,CPU contention and sleep can change throughput. Full epoch QC,validation,checkpoint writes and subsequent independent evaluation are extra. The mechanical TRAIN-only pair projection is2h40min; a cautious planning estimate including validation/replay/overhead is **3–5 awake hours after launch**, not an ETA for allstep3 or a guarantee.

Separate no-model verifier recomputed shared config/initial/data parity,two complete no-drop schedules719updates/epoch,disabled calibration/test mappings,exact timing counters/means and scalar projections. Source/protocol/config hashes and ordinary CLI candidate rejection flag remain bound. Original190 technical artifacts unchanged. The configs are still `candidate_not_launch_sealed`: final exact frozen-trainer integration verification and a separate scoped execution release/idle guard remain before quality training. There is no quality training running merely because timing passed. Fixed endpoints/seeds/horizon are no longer unspecified; outstanding work is now integration and release, not another redesign. See CURRENT_GATE_LEDGER_V3_20.json.
