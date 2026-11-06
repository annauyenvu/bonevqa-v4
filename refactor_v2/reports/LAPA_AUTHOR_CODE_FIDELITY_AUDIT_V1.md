# LaPA author-code fidelity audit — preliminary, 25 October 2026

No new training release. Current P diagnostic remains unchanged. Latest human priority: PAPER_FIDELITY_POLICY_20261025.md.

## Source identity

Author repository: https://github.com/GaryGuTC/LaPA_model . A read-only git ls-remote resolved main to 677faf90c010dac133dcbde516bb7c96448b29d1 on this review. Source observations below were retrieved from browser-visible main/raw pages; immutable byte snapshots and hash verification against that commit remain pending. Do not treat cached page contents as already commit-verified source.

## Main model routing

Source: https://github.com/GaryGuTC/LaPA_model/blob/main/m3ae/modules/m3ae_module.py

Observed: 32 prompts, normal initialization standard deviation 1/hidden_size; answer token embeddings projected along their sequence dimension to 128; prompt LF repeated alongside multimodal layers. PF expects 577 graph elements and learned Conv1d reduction to four graph slots for 32 prompts. A second Conv1d pools concatenated prompt/graph slots. Final representation weights latent/text/image by 1/0.1/0.1; eta is 0.1. Local code instead uses a TRAIN answer-level pooled bank, standard deviation0.02, one LF sequence and a baseline-preserving gate; no active PF. These are substantive port differences. The prepared409-node graph cannot replace the577-element asset silently.

## Attention and GAT implementation

Source: https://github.com/GaryGuTC/LaPA_model/blob/main/m3ae/modules/language_encoders/bert_model.py

Observed LF order: language, image, concatenated multimodal features, then feed-forward. PF uses graph features as attention queries and prompts as keys/values. GAT_module sets GAT_head=1 while its cross-attention defaults to8heads: earlier shorthand “8-head GAT” is NOT an accurate description of this released code. GraphAttentionLayer replaces non-edge logits with zero and normalizes along dim1, rather than applying a negative-infinity non-edge mask. This may admit non-edge attention; a numerical fixture and paper/code discrepancy resolution are required. Existing local8head masked GAT is not an exact copy. Do not silently fix released behavior and label the result exact reproduction; distinguish author-code and paper-intended variants if needed.

## Loss semantics

Source: https://github.com/GaryGuTC/LaPA_model/blob/main/m3ae/modules/objectives.py

Released VQA loss scales mean BCE by answer-class count and adds eta times the SUM of per-sample cosine losses. Its consistency target averages raw answer word embeddings. Local consistency averages over batch and masked contextual answer tokens, alongside different answer/fracture/OPEN objectives. Eta alone therefore cannot make loss scales equivalent. Audit normalization, padding, effective batch size and language-embedding updates before choosing a paper-aligned config. Current eta0.01 diagnostic is historical, not source-faithful.

## Backbone and launch configuration

Sources: https://github.com/GaryGuTC/LaPA_model/blob/main/run_scripts/train.sh and https://github.com/GaryGuTC/LaPA_model/blob/main/m3ae/config.py

Released train.sh selects clip16, text_roberta,384px, M3AE checkpoint and dataset-specific answer assets. Thus the paper's reported backbone and repository launch recipe must not be conflated. Named configs and script overrides require full resolution; reading defaults alone is insufficient. No local matching checkpoint was identified by the limited .evaluation_runtime filename search (this is not proof of global absence).

## Required before reference release

1. Retrieve immutable source files with hashes; verify redistribution/license and dependencies without changing runtime.
2. Verify author external_data graph/node token order, answer preprocessing and upstream checkpoint provenance. README links external data/checkpoints, M3AE weights and RoBERTa assets; accessibility and local availability are not yet verified.
3. Add synthetic routing/loss-reduction/graph-mask fixtures. Resolve paper-versus-code conflicts explicitly.
4. Separate native author-code reproduction from shared-BoneVQA mechanism port, and freeze bounded protocol/config and matched controls before empirical gates.

No claim of faithful PF, original LaPA performance, new QA benefit or independent clinical confirmation follows from this audit.
