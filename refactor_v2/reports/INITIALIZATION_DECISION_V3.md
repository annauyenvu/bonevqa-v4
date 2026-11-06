# Initialization decision for the revised data protocol

Status: decision recorded; replacement checkpoint not yet trained or sealed.

The historical stage2_pmcvqa_full checkpoint is ineligible as the common initialization for confirmatory experiments. Its payload does not identify the training manifest, and train_lon explicitly mixes historical PMC training and test records. This is insufficient provenance; it does not prove that this particular checkpoint trained on test.

Use the cached published BiomedCLIP and Qwen base weights to build a fresh common initialization, followed by alignment on an audited ROCO training manifest and VQA training on an audited PMC training manifest. Record exact base revisions, file hashes, train manifest hashes and shared initialization tensor hashes before any ablation. New randomly initialized fusion and heads require actual alignment/training and cannot be treated as a pretrained VQA model.

Exclude evaluation-linked components from all upstream training, including ROCO alignment. Same-pHash matches alone do not complete the cross-dataset audit; unequal-pHash neighbours, source article grouping, and image decoding must be covered before sealing the pretraining manifests. Keep PMC train_lon out of the revised pipeline.

FracAtlas component CV manifests are development-informed evaluation. Each outer fold requires its own detector trained using only that fold's training partition; model selection uses model_val and probability calibration uses calibration. Outer test is evaluated after the inner choices are frozen. This does not establish an untouched external clinical validation, and known test-informed development must be disclosed.

LaPA scope is GM and LF. No PF graph is required or claimed.
