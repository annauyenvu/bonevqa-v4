#!/usr/bin/env bash
# Clean-clone test using ONLY the repository's own tools/relocate.py (as documented in its README).
#  1. export the committed tree (no .git, no untracked files) to a fresh directory at a DIFFERENT path;
#  2. rewrite the original absolute root with the repository's own tools/relocate.py (README, Setup);
#  3. check that every input the paper-analysis scripts need exists;
#  4. run the paper-analysis commands and compare each regenerated result file with the committed one.
# Usage: tools/repro_check.sh <repo> <workdir> <python>   (workdir is deleted and recreated)
set -u
REPO=$1; WORK=$2; PY=$3; HERE=$(cd "$(dirname "$0")" && pwd)
rm -rf "$WORK"; mkdir -p "$WORK/clone"
git -C "$REPO" archive HEAD | tar -x -C "$WORK/clone"
echo "commit: $(git -C "$REPO" rev-parse --short HEAD)  ->  $WORK/clone"
( cd "$WORK/clone" && "$PY" tools/relocate.py )
echo "remaining author-path literals in .py: $(grep -rl '/Users/uyen/Downloads/BONEVQA' --include='*.py' "$WORK/clone" | wc -l | tr -d ' ')"

cd "$WORK/clone/refactor_v2"
# input check: the stored predictions and split files the analyses read
need=(../.evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json reports/local_noaux_refinement_v1/fold_0/B/test_predictions.json
      reports/v5_studies/access_v1/fold_4/V4_placebo_seed43/test_predictions.json reports/v5_studies/access_v1_methods/fold_4/L_seed42/test_predictions.json
      reports/access_v1/routing/fold_4/V4/new5_router.json reports/access_v1/external)
miss=0; for f in "${need[@]}"; do [ -e "$f" ] || { echo "MISSING input: $f"; miss=1; }; done
[ $miss = 0 ] && echo "inputs: all present"

run() {  # name, stored result (relative to refactor_v2), command...
  local name=$1 out=$2; shift 2
  cp "$out" "$WORK/$name.stored.json" 2>/dev/null || { echo "[$name] no stored file $out"; return; }
  if "$@" > "$WORK/$name.log" 2>&1; then
    printf '[%s] ' "$name"; "$PY" "$HERE/compare_json.py" "$WORK/$name.stored.json" "$out" && echo "   -> REPRODUCED" || echo "   -> DIFFERS"
  else echo "[$name] FAILED (see $WORK/$name.log)"; tail -3 "$WORK/$name.log"; fi
}
export PYTHONPATH=.
run stats      reports/access_v1/stats.json                  "$PY" -m experiments.access_stats_v1
run contrasts  reports/access_v1/contrasts.json              "$PY" -m experiments.access_contrasts_v1 contrasts
run methods    reports/access_v1/methods.json                "$PY" -m experiments.access_contrasts_v1 methods
run routing    reports/access_v1/routing/results.json        "$PY" -m experiments.access_routing_v1 analyze
run paraphrase reports/access_v1/paraphrase/results.json     "$PY" -m experiments.access_paraphrase_analyze_v1
run external   reports/access_v1/external/results.json       "$PY" -m experiments.access_external_v1 analyze
run det_ext    reports/v4_detector_ext_v1/v4arm_full.json    "$PY" -m experiments.v4_consistency_v4arm_v1
run availability reports/access_v1/availability.json         "$PY" -m experiments.access_availability_v1
run review2    reports/access_v1/review2.json                "$PY" -m experiments.access_review2_v1
run review3    reports/access_v1/review3/posthoc.json        "$PY" experiments/access_review3_v1.py .. reports/access_v1/review3/posthoc.json reports/access_v1/review3/detector_scores.json
