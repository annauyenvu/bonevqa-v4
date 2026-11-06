# PF with reconstructed SLAKE graph

User approved this reconstruction on 26 October 2026. Implementation:
`experiments/pf_slake_reference_v1.py`. Historical source and graph seals remain unchanged.

## Fixed mapping

- Verify all existing graph artifact hashes before loading.
- Preserve the 409 ordered node records (107 organs, 302 diseases) and 304 source medical relations; do not pad to 577 or invent edges.
- Tokenize exact stored English labels with the locally downloaded author-linked RoBERTa tokenizer, no special tokens, no truncation. There are 2002 nonpadding tokens; maximum 13 per node.
- Represent each node by the masked arithmetic mean of raw word embeddings. This is a disclosed adaptation: the missing author's organ_disease_info.pt representation is not established.
- Caller supplies the backbone's word embedding module; this adapter does not load weights or freeze/train the backbone implicitly.
- Use pinned author GAT_module without changing its nonedge behavior. Graph queries attend to 32 LF prompt keys/values. Pool 409 nodes to four slots and concatenate with 32 prompts, then pool 36 slots.
- Modes real/rewired/bypass share tensor dimensions and modules. Rewired uses the existing degree-preserving control. Bypass skips GAT but retains node-to-prompt attention and four slots. Bypass has fewer active parameters/operations, so it is not a compute-matched structural control; real versus rewired is the principal structural comparison.

## Executed component check

Existing offline Python environment, CPU8, seed42, actual offline RoBERTa tokenizer, random width32 embeddings/prompts, batch2, evaluation mode. All three outputs finite and shape2x32. Backward reached prompt input, word embeddings and GAT output weights. Strict same-process state reload reproduced outputs exactly.

Real-minus-rewired maximum absolute difference was 2.384185791015625e-7; real-minus-bypass 0.0009920299053192139. The very small rewiring difference is not evidence of meaningful graph sensitivity. A controlled intervention test and actual-width real-image integration are still required. No optimizer, pretrained weight loading, fresh-process replay or VQA quality run occurred in this check.

## Remaining integration gates before quality release

Bind node word embeddings to the selected actual backbone and validate its tokenizer compatibility; integrate after the sixth interleaved LF call; verify controlled graph/node interventions and answer-loss gradient routing; match pre-update initialization across controls; perform disposable real-image updates, frozen-weight invariance, target-free public inference and fresh-process full-state replay. Freeze a bounded evaluation protocol before any quality training. Existing completed pilot scores must not be presented as PF results.

Label: PF following LaPA's mechanism with reconstructed SLAKE graph. Native LaPA reproduction remains unverified.
