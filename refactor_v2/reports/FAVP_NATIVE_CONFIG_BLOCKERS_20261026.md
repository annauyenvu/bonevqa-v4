# FAVP native configuration acceptance

Static review of pinned commit b43941bac4774b796fd8650a263ea32c220e6aba, 26 October 2026. No execution, downloads or changes to author source.

Assets alone do not make the released configuration runnable or establish initialization equivalence.

- `favp/configs/models/vicuna0.yaml` leaves `llama_model` empty. Resolve the exact Vicuna V0 7B snapshot/tokenizer and record hashes.
- `train_configs/train_rad.yaml` leaves `q_former_model`, `ckpt`, output directory and `max_epoch` empty. The Q-Former constructor calls `load_from_pretrained`; the loader rejects a path that is neither a URL nor an existing file. A supplied checkpoint's role must be explicit, not inferred from its name.
- README describes stage-1 and stage-2 pretraining followed by downstream fine-tuning, and labels the downloadable alternative a VQA-RAD checkpoint. That downstream checkpoint must not be represented as a clean pretrained initialization without provenance review.
- `run_scripts/train_rad.sh`, read directly from the pinned Git object (not present in the sparse working checkout), only selects the RAD YAML; it does not fill the empty settings above.
- `run_scripts/stage2_pretrain.sh`, also read from the Git object, selects `stage2_finetune.yaml`, whereas the inspected README points to `stage2_pretrain.yaml`. Resolve the intended file and stage handoff explicitly before execution. This discrepancy is not proof about the authors' actual experimental environment.
- Native SAM-Med2D, EVA, Q-Former and Vicuna assets, plus the pretraining-stage checkpoint chain, need separately identified provenance. Confirm missing/unexpected state keys rather than accepting `strict=False` as verification.

Required receipt before native gates: resolved effective configuration; every asset path/hash/revision; stage and dataset provenance; trainable/frozen parameter inventory; tokenizer and preprocessing; bounded epoch/update budget; independent evaluation-role binding. Do not infer missing epoch counts from BoneVQA validation scores. Any corrected launcher or filled configuration is a disclosed adaptation, not an unchanged released command.

No native quality run is released by this document. Existing shared-backbone and historical experiment seals remain untouched.
