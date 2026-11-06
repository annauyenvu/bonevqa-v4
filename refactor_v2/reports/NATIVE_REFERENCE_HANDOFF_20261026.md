# Native reference readiness after shared-backbone pilot closure

The shared-backbone GM/LF experiment is closed in lapa_shared_quality_v1/closure_seal.json. Its outcome does not certify native LaPA, FAVP or Localization Lens. No quality job remains active at this check.

## Concrete next dependency: native LaPA assets

The pinned author config names download/external_data/adj_matrix.pt and download/external_data/organ_disease_info.pt. These must arrive as a matched pair with the 577-node ordering and tokenization provenance. The existing independently reconstructed 409-node SLAKE graph cannot establish equivalence. Record source URL, revision where available, file size and SHA-256 before deserialization; inspect unknown serialized assets in an isolated safe loading path.

The author README separately points to pretrained M3AE and RoBERTa assets. A downstream RAD/SLAKE-finetuned checkpoint is not a clean pretrained initialization. Establish checkpoint training datasets and label exposure before accepting it. Pin the actual resolved launch configuration rather than combining base config defaults with paper statements.

## Acceptance package

For each asset retain an immutable receipt with original source, retrieval time, size, checksum, license/usage information, tensor keys/shapes, tokenizer revision and ordering where relevant, and pretraining exposure. A successful HTTP response containing an error page is not an asset. Do not infer access restrictions or a dead link solely from prior retrieval failures.

When the asset package is verified, validate the author route in a separate controlled environment or explicitly document any compatible-runtime adaptation. The established offline experiment environment remains unchanged. First run bounded forward/backward/reload gates and estimate resource use; do not launch an original-backbone quality experiment from source availability alone.

FAVP additionally needs its matching SAM-Med2D, EVA-CLIP/Q-Former and Vicuna assets. Lens needs its actual SmolVLM and SAM assets plus valid existing supervision for the selected view protocol. Generic organ fallback labels and test-as-validation code paths are not acceptable data adaptations.

Native execution is currently blocked by unverified assets/provenance, not by waiting for another shared-backbone training run. No new large model download, dependency installation or training has been performed in this readiness check. Source-only audits and asset identification remain possible; any future acquisition must respect the user's existing offline-runtime and bounded-work constraints.
