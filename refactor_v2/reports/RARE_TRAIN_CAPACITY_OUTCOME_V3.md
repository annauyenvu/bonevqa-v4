# Tiny rare-TRAIN capacity gate — 2026-10-21 02:57 ICT

The real model can fit this fixed16-row fixture under the predeclared64-update cap. Both controls passed. This closes a scoped technical capacity question, NOT generalization, visual grounding or clinical quality.

| Disposable updates | Native total loss: correct /16 | Answer-only loss: correct /16 |
|---|---:|---:|
| 0 | 3 | 3 |
| 16 | 12 | 12 |
| 32 | 16 | 16 |
| 64, primary endpoint | 16 | 16 |

Teacher-forced CE, averaged over the same four fixed evaluation batches, was0.470923initially for both. LAST64 native0.000796,answer-only0.001497. These are TRAIN diagnostic losses, not calibration or independent efficacy metrics. Selected QFormer-query/prefix/VisualLoRA/LLMLoRA answer gradients were finite/nonzero at1,32,64 and each tracked tensor changed. The report does not assert every parameter had a gradient.

Protocol SHA42e738302c2f56b1148e9dabf87b9fcded23b10c8144517ded7c4ba2ef2386f3 locked16existingrareOPEN TRAIN rows/16components, sorted-question round-robin without prediction-based selection:7description,7location,1count,1region. Both controls warm-started the SAME already task-trained rawB LAST296 SHA94594edf64f8ef3fd7ae974e290b86900e186661d03c09e68d90e34c51ff3ddf; all263initial tensors and initial16predictions/CE exactly equal. This was NOT a new clean PREtask initialization or independent VQA training run.

Each control used64batch4 AdamW updates,lr1e-4,betas(.9,.98),wd.01,clip1,constantLR; every selected row appeared16times. Batch order,epoch-seeded RNG policy, parameters and input rows identical. Native total versus exposed original loss_answer was changed only inside this isolated technical experiment; originalruntime/production loss and configuration remain unchanged. Snapshot32success did not trigger earlystop or checkpointselection: both finished64. No budget increase/retry.

All public decoded answers used originalimage/rawquestion only, no GTtype/region/annotationROI. ExistingGT entered private TRAIN teacher-forced loss only. No V/L/P/T,loss-weighting or routing adapter, no inventedlabels, no validation/calibration/test inference. No independent image-grounding control was run here, so memorization must not be called clinical grounding.

Total245.801sec4min5.801sec,128disposable optimizerupdates. ALL resulting weights discarded and no `.pt` checkpoint saved. Sourcecheckpoint unchanged. Separate no-model audit checked source/protocol/checkpoint hashes,16identities at every snapshot,decoded-correct arithmetic,finite CE/selected gradients andupdates,matchinginitialsnapshots,exact64updatebudget and absence ofsavedweights. It does NOT independently replay training or a trainedmodel: discarded weights cannot be reloaded. Report `rare_TRAIN_capacity_v3_1/report.json` SHAc98f4a29666983cf4906ae6f803d06cda832b4481ba2a640d6a55102342189ed; `independent_artifact_audit.json` passed. Original190technicalartifactseal verified unchanged; finalps showed noactive model/training jobs.

Interpretation: a fatal decoder/answer-gradient path failure or inability to fit these16rare targets is not supported. Native auxiliary losses did not prevent fitting this fixture; this result does not justify deleting them. Earlier TRAINtoken audit excludes configuredQwen truncation on534OPENrows. The full TRAIN rarity checkpoint still missed58/72rare answers, and independent development quality still has no established benefit (114vs113of142,pairedintervals include0). Different TRAIN exposure/optimization may merit a separately controlled hypothesis, but this tiny fit neither proves that more full-data epochs will help nor releases any new training. No automatic sampling/LR/loss change, longtraining,FinalSolution or production promotion. Fullstep3/sevenstepplan remain incomplete.
