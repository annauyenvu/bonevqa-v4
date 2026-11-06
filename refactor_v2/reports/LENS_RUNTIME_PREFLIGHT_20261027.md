# Lens static runtime preflight — no Lens model run yet

Prepared while the immutable FAVP pilot/replay was running; sequential method execution remains FAVP closure before Lens tests/training. No package installation or runtime change was made.

Observed in the existing offline runtime: Transformers4.40.2 has AutoProcessor, but does not expose Idefics3ForConditionalGeneration, SmolVLMForConditionalGeneration or AutoModelForImageTextToText. The downloaded SmolVLM256M config declares model_type`idefics3` and architecture`Idefics3ForConditionalGeneration`. The pinned Lens training.py/inference.py explicitly import that class. Therefore the current environment cannot run the native Lens route unchanged; possession of safetensors alone is not model acceptance.

The actual pinned author pyproject.toml requests torch>=2.2, transformers>=5.0, accelerate>=1.2 and peft>=0.14 for its train extra. This is a source-declared requirement, not empirical compatibility or a measured minimum. Do not guess a lower Transformers version merely because a model architecture existed there.

After FAVP closure, establish an isolated, version-bound Lens runtime (or an explicitly audited source port) without changing the historical/FAVP environment or sealed dependencies. Resolve compatible package versions and storage before installation, preserve8GiB reserve, and do not install a large dependency stack blindly. Keep downloaded model assets local/offline during acceptance and avoid unrelated network/model downloads.

Then verify the actual SmolVLM processor's per-view tile counts, image/token/attention mapping and source pooling on real original plus valid existing-supervision views. Do not assume equal tiles from shape alone. Correct observed faults in versioned adapters, not immutable author source. Verify DCL/InfoNCE active batches and gradients, original-only public inference, fresh reload, no organ-label fallback invention and correct TRAIN/model-validation roles before bounded Lens training. Prior helper-only fixtures are not a native model release.
