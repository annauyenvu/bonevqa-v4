# SAM-Med2D acquisition metadata

26 October 2026, metadata-only check; no checkpoint downloaded in this check.

The official SAM-Med2D repository Results table links its SAM-Med2D 256 model to https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view . The Drive viewer identifies `sam-med2d_b.pth` and lists 2,561,829,701 bytes (about 2.562 GB). Source: https://github.com/OpenGVLab/SAM-Med2D . This supersedes the previously unknown size, but is not a downloaded checksum or a tensor-compatibility verification.

Known FAVP asset subtotal including this checkpoint is 18,497,526,182 bytes (about 18.498 GB), excluding stage-1/2 handoff checkpoints and small tokenizer/config assets. Local free storage measured about 38 GiB after LaPA downloads. Do not interpret file capacity alone as sufficient working space for native FAVP training or install all assets automatically without the remaining provenance/storage checks.

Existing MedSAM-base is not a substitute. The unexpectedly larger file relative to a plain SAM-base weight is not grounds to strip optimizer/training state or guess its contents; any later serialization inspection requires a safe, separate procedure. No native training release.
