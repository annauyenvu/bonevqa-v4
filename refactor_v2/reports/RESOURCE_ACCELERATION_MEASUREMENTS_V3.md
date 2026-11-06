# Resource acceleration measurements

User requested maximum practical throughput, not merely maximum utilization. Read-only hardware inspection found12CPU cores (8performance/4efficiency),24GiBRAM, no swapin/swapout counters and no recorded thermal/performance warnings at the snapshot. No model worker was active before the benchmarks. Those observations are snapshots, not a guarantee of future memory/thermal headroom.

Current fixed mixed4TRAIN-row native forward/backward benchmark used ONE resident LAST model, one warmup per thread setting and two reverse-order measured repeats. It created no optimizer and preserved all263trainable-state hashes and checkpoint bytes.

| CPU threads | Median seconds/batch |
| --- | ---: |
|8|0.601262|
|10|0.645934|
|12|0.678055|

All selected-gradient/loss numerical screens passed.12threads is12.77%slower than8 here, so forcing all12cores is not the measured fastest configuration. Interop1 was fixed in this test; there is no measured comparison against the old default interop setting. Four repeated rows and two timed repeats do not determine full-run sustained throughput.

Separate CPU/MPS benchmark on the same fixture found CPU0.578701seconds/batch versus MPS0.379128, a1.5264×throughput ratio. No parameters/checkpoint were changed. MPS gradients were finite but did not match the selected cross-device screen: e.g. loss delta0.001697, fracture-head selected max gradient delta0.128338. Because both passes used TRAIN-mode dropout, differing CPU/MPS RNG masks can explain differences; this is not an isolated numerical-fault diagnosis. The existing installed-runtime MPS RNG-state-restore replay probe also failed. GPU quality training remains unreleased until a properly scoped GPU-specific deterministic/update/restart/replay gate passes; the speedup is not a full-pipeline guarantee.

Applied workflow policy is recorded in FAST_EXECUTION_POLICY_V3.json: measured8-thread CPU choice; one resident model per related diagnostic workload; reuse safe in-memory fixtures; run dependent safe stages consecutively instead of waiting for reporting cadence; retain a single required fresh-process replay at a completed stage boundary. Avoid competing memory-heavy models or full-resolution image caches without measurements. No network, library installation, power settings, new quality training, clinical labels or scientific endpoints were changed.
