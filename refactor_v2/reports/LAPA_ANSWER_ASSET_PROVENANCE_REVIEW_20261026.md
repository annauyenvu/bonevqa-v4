# LaPA answer assets: provenance distinction

Read-only follow-up, 26 October 2026. No training, dependency edits, asset download or author contact.

## Public evidence

In [issue 6, owner comment](https://github.com/GaryGuTC/LaPA_model/issues/6#issuecomment-2141178212), the author describes constructing `ans_tokenizer_dict` from answer tokens across training, validation and test splits. A [second owner comment](https://github.com/GaryGuTC/LaPA_model/issues/6#issuecomment-2141769629) specifies the RoBERTa-base tokenizer. This is a description of the lookup dictionary, not verified provenance of the distinct GM bank artifact.

The returned comments on issue 14 contain another user's request for the External Data link, not a verified replacement asset URL. Reports of inaccessible links do not by themselves establish that access is permanently unavailable.

## Trace in pinned author source

Revision: `677faf90c010dac133dcbde516bb7c96448b29d1`.

- `m3ae/modules/m3ae_module.py:108` loads `ans_token_path`. Lines 234–238 embed this tensor, expand it across the batch, apply the sequence projection and feed it to the prompt generator. Its actual contents and split provenance remain unverified.
- `m3ae/modules/objectives.py:31–36` separately loads `ans_tokenizer_dict_path`, selects the current batch's answer entry, embeds it and takes its token mean as the consistency target.
- In `compute_vqa`, inference and logits are computed before this lookup. The inspected objective does not pass that target embedding into the logits computation. A cross-split lookup dictionary therefore is not, by itself, evidence of test-answer-conditioned inference or test answers entering training batches.
- The launcher explicitly supplies two separate paths. Neither their similar names nor their coexistence establishes identical contents.

## Consequences

Do not claim that the published method or results leak test answers based only on this comment. Inspect the actual GM bank, generation procedure, batch roles and selection protocol before drawing that conclusion. Conversely, do not claim TRAIN-only native bank provenance while its bytes are unavailable.

The BoneVQA shared-backbone pilot retains its disclosed TRAIN-only bank and training-target construction. Historical sealed results are unchanged. Any native reference must document each asset separately: bank tokens and ordering, target lookup construction, tokenizer revision, padding, special tokens, split provenance and hashes. Missing native assets still block exact-reproduction claims; this review does not authorize substitute assets or new quality runs.
