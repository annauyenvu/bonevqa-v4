# Fixed TRAIN automatic-view coverage check

Before observing new proposal outputs, use exactly the four historical TRAIN gate indices in `reports/vnext_local_tokens_v1/gate_indices.json` (8, 5, 16, 26), in that order, with no additional image search. These are an interface-development subset, not representative quality evaluation. Reuse the existing IMG0000044 proposal receipt only for comparison; run the fixed subset consistently.

Use original images only, native SAMMed2D and unchanged RAD thresholds/grid, CPU point batch 16, native circle renderer and pretrained EVA/QFormer. No answer, boxes or clinical labels are used by the route. Report every case including zero proposals; never tune proposal thresholds or manufacture local views to make the gate pass.

For each case record accepted proposal count and time, rendered view count, expected memory length 257×(1+proposal count), finite/exact-repeat query outputs, and native local-before-global token order. When natural local views exist, compare their queries against original-only input as a technical intervention (not QA improvement). Save compact query features only; do not store full-resolution masks or EVA memory. If a case exceeds 32 proposals, record a resource-limited route rather than silently truncate or claim full acceptance. This bound is a technical execution guard, not a quality configuration.

This check does not train the model, prove stage-1/2 alignment provenance, or release FAVP quality training. Any unavailable local-branch evidence stays explicitly pending.
