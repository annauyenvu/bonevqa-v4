# Native asset acquisition progress, 26 October 2026

Final update: all accessible selected packages completed and passed a full byte audit: 36 files, 23,651,792,013 bytes. See `reports/native_asset_acquisition_closure_v1/RESULTS.md` and sealed `inventory.json`. The historical Started section below records chronology; no downloader remains active.

Approved asset downloads only. Model environments and quality training are not released by this receipt summary.

## Completed

- LaPA: author-linked M3AE 1,566,863,121 bytes, seven selected RoBERTa files. Local hashes and expected sizes retained under `references/native_assets_lapa_v1`.
- FAVP: Q-Former 433,481,467 bytes; EVA 2,025,249,237 bytes; SAM-Med2D 2,561,829,701 bytes. Each has a local SHA-256 receipt under `references/native_assets_favp_v1`.
- FAVP Vicuna: eight files, 13,477,494,841 total bytes including tokenizer/config. Revision `0abeb8872d75421e58ec146c39b31f5dc78c7613`; weight and tokenizer LFS SHA-256 matched publisher metadata; small-file Git blob IDs matched. Receipt `references/native_assets_favp_v1/vicuna-7b/receipt.json` has complete=true. This is stronger byte provenance than a locally computed hash alone; it does not establish FAVP stage handoff weights.

## Started: Lens

Sequential byte downloader `experiments/download_native_lens_packages_v1.py`, exec session 87602. Inspect receipts/progress/failure and live process before resuming; do not duplicate.

- SmolVLM-256M-Instruct revision `7e3e67edbbed1bf9888184d9df282b700a323964`, selected root-level weights/config/tokenizer/README.
- SAM-Huge revision `87aecf0df4ce6b30cd7de76e87673c49644bdf67`, selected safetensors/config/processor/README.
- Destination `references/native_assets_lens_v1`; publisher SHA-256/Git blob verification required, no duplicate ONNX, TensorFlow or PyTorch-bin weights. Each package enforces 20 GiB disk reserve before download. Latest free space before Lens was approximately 41 GiB.

## Still unresolved

LaPA External Data link explicitly expired: graph577/node order and associated answer assets remain unavailable. No author contact performed. FAVP stage1/2 checkpoint provenance and effective configuration remain unresolved. Lens source-to-paper route, valid supervision and processor tile-count contract still need model-level evidence. No checkpoint has been deserialized by these download scripts; native-method reproduction is not complete.
