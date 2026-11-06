# FAVP input-route acceptance checklist

Static inspection, 26 October 2026. Author revision b43941bac4774b796fd8650a263ea32c220e6aba. No source modifications, model execution, downloads or training release.

## Observed route

`favp/models/favp.py`, `get_masks`, converts automatic generator XYWH boxes to XYXY. `encode_sam_out` reads only `image_path[0]` through OpenCV. It passes that array directly to mask generation, then reverses its channel order for the visual-prompt renderer. This exposes a BGR/RGB boundary requiring an explicit fixture; it is not evidence of measured performance damage.

The renderer first appends the original image, then one rendered full-image view per proposal. These are not necessarily cropped regions. `encode_sam_out` applies the shared visual encoder to each view, concatenates proposal-view tokens first and original-view tokens last, then invokes Q-Former and the language-model projection. The zero-proposal branch encodes the original alone. The source includes hardcoded CUDA transfers.

Released RAD training/testing YAML selects `circle`; SLAKE training selects `mask`. All three SAM-enabled dataset configurations specify batch size 1, consistent with the first-path implementation. Do not silently use a multi-image batch or select a prompt style after examining evaluation scores.

## Required bounded checks before adaptation release

1. RGB sentinel image: trace decoded channels, generator input and rendered image separately. Preserve the author route as a named reference; any color correction must be disclosed as an adaptation.
2. Zero/one/multiple synthetic proposals: check original-first rendering, proposal count, XYWH conversion and local-first/global-last token order. Synthetic masks are computational fixtures, not clinical labels.
3. Two distinct image paths: public adapter must reject unsupported batching or explicitly process every sample; never silently reuse the first image. Confirm batch-1 parity with the pinned method.
4. Lock dataset-to-prompt mapping and proposal settings prospectively. No threshold or style search on assessment data.
5. Verify SAM-Med2D, EVA, Q-Former and language-model asset hashes/provenance before real-model gates. Existing MedSAM weights are not an equivalent substitution.
6. After assets are available, test original-only public inputs, deterministic replay, frozen-module invariance and intended gradients/updates with a bounded gate before any quality run.

These are static findings and proposed acceptance checks, not completed empirical gates. They do not establish full paper equivalence. Historical sealed BoneVQA runs remain unchanged.
