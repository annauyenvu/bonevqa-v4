# Fixed-checkpoint TRAIN diagnosis — 2026-10-21 02:22 ICT

Both1173-row TRAIN passes and fresh-process90211 replay of90210 predictions completed in931.830sec (15min31.830sec), with every answer/probability/actual route exact. No weights, labels, templates, thresholds or optimizer state changed. No model-validation/calibration/test inference in this diagnosis.

| Frozen LAST296 | TRAIN correct /1173 | OPEN correct /534 | Globally rare-answer correct /72 |
|---|---:|---:|---:|
| Raw B | 943 | 404 | 10 |
| Uniform sequence loss | 941 | 411 | 10 |
| Rarity-weighted sequence loss | 951 | 410 | 14 |

The rarity arm learns four more rare-answer TRAIN rows net, but still fails58/72. RawB→rarity rare transitions:9bothcorrect,1correct→wrong,5wrong→correct,57bothwrong. Acrossall1173:922bothcorrect,21correct→wrong,29wrong→correct,201bothwrong. Uniform rare transitions:7bothcorrect,3correct→wrong,3wrong→correct,59bothwrong. These are descriptive TRAIN changes, not independent effect estimates or proof that imbalance is the sole cause. GlobalTRAINanswerfrequency<=5 is the previously fixed diagnostic stratum; it is NOT identical to question-conditioned loss weights.

Actual routes match the existingGT CLOSED/OPENtype for all1173 in both loss arms. All38hardware questions remain correct in both. The rarity TRAIN count-question correct40/58 versus rawB36/58, but location correct19/32 versus21/32 and description90/168 versus88/168; changes are mixed. Do not choose a production recipe by these subgroups. Literal string-overlap categories do not establish clinical equivalence or permit relabeling.

Separate no-model artifact audit independently rechecked source/protocol/checkpoint/prediction/replay hashes,1173record identities, every summary/native metric and rare stratum, rawBreference and transition counts. Report `B_loss_TRAIN_diagnosis_v3_1/report.json` SHA affa28fb9f99f6c04c6601785bdf80e4d9b5ebf8b0cdb3a42f452601c1f5ccec; audit `independent_artifact_audit.json` passed. The frozen B reference was not reinferred.

Tokenizer-only follow-up used the actual cached Qwen tokenizer and original rawB prompt formatter, existingTRAIN targets with leading space as in training, add_special_tokensFalse, max_answer_tokens32/max_new_tokens32/prompt_limit56. All534OPEN targets fit; longest17tokens, zero truncated targets, zero text-plus-EOS budget overflow, zero prompt overflow among1173questions, and all normalized full decodes match existingGT. Cached tokenizer/config/source/TRAIN hashes recorded and rechecked. Report `TRAIN_token_budget_v3_1.json` SHA8647a4a6395d9680260e684c3070d64e38d1f411984eee5ea4bffe2ce739cd92. No neural model was instantiated for tokenizer audit. This excludes configured Qwen text truncation for these TRAIN rows, not every possible preprocessing/capacity/clinical issue. No evidence supports increasing text limits as a fix here.

Earlier independent development outcome remains unchanged: rarity114/142 versusB113/142; paired accuracy/AUROC/AP intervals include0, no established benefit. TRAIN improvements do not supersede that conclusion. Both training pilots are complete; there are no pending epochs. No longtraining or loss promotion, no novel clinical labels/annotationROI/PF, no FinalSolution/production release. A possible next technical question is whether the exact real model can fit a predeclared small rare-TRAIN fixture with live answer gradients; that would require its own bounded protocol and would still not be clinical evidence. It has NOT been launched or claimed passed here.
