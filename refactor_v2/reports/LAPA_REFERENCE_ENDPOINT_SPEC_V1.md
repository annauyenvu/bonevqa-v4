# Reference endpoint specification, 25 October 2026

The next real-record technical gate uses the released LaPA classifier layout: Linear(768,1536), LayerNorm, GELU, Linear(1536,41), with author initialization. The earlier real-image gradient receipt used a disposable single linear head and remains valid only for that head. It does not certify this endpoint.

The fixed TRAIN vocabulary supplies the 41 answer classes. Existing single ground-truth answers are encoded with one-hot scores. Evaluation answers absent from this vocabulary remain incorrect/unsupported; they do not extend the vocabulary. Report vocabulary coverage and closed-vocabulary accuracy separately from earlier generative VQA results.

The source objective is mean binary cross entropy multiplied by class count, plus eta=0.1 times the batch SUM of cosine distance between GM mean prompts and the unmasked mean of raw target-answer word embeddings. Target tokenization, special tokens, padding length and embedding identity must be recorded before running the gate. Padding is included in the author-style mean; any masked alternative is a separate extension.

The endpoint implementation is experiments/lapa_reference_endpoint_v1.py. Its classifier layout and objective were checked against the pinned author's m3ae_module.py lines156–162 and objectives.py lines22–42. The implementation is not yet empirically gated with the full real-image runner.

Next gate: two disposable updates on the same TRAIN examples, explicit answer/consistency gradient routing, finite deltas, frozen-weight invariance and fresh-process reload. Use a separately saved initialization and receipt. The task answer must never be an input to public inference. This specification authorizes no quality run, performance selection, threshold search or additional seeds.

The shared BiomedCLIP/PubMedBERT backbone and random cross-modal initialization remain departures from native LaPA/M3AE. PF and exact native assets remain unresolved. A matched classifier baseline and prompt controls are required before interpreting accuracy as a prompt contribution.
