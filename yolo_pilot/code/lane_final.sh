#!/bin/zsh
# Final configuration of the thesis on folds 0-3, scope-freeze version (user 20/11 17:0x; deadline: tables before
# 08:00 21/11).  JF = J recipe (V4-CF + joint YOLOv8 proposals + BTXRD QA, train_joint, seed 42, CPU) trained on every
# fold with the SAME proposals: FracAtlas level 0.02 (crossfit_yolo_joint_sweep/r0.02/fold_<f>/proposals_fold<f>.json)
# and BTXRD level 0.02 (r0.02/btxrd_fold<f>/proposals_btxrd_insample.json).  Name JF_v4cf_yolojoint_btxrd_r0.02.
#   1. BTXRD threshold, inference only (no retraining): the trained JF fold 0 is evaluated on BTXRD model_val with the
#      BTXRD proposals at normal-image rates 0.02 / 0.05 / 0.10 / 0.15 / 0.30 (code/build_btxrd_proposals.py <f> <l>);
#      the level with the highest tumour AUROC on model_val is chosen once and locked in
#      v4_grounded_next/runs/joint/final_btxrd_level.json; test and external are not looked at.  It is then applied
#      blind at inference on every fold (training proposals unchanged).
#   2. per fold 0 -> 1 -> 2 -> 3, one at a time: train, FracAtlas test + calibration, BTXRD model_val/test/external at
#      the locked level, comparisons (vs A, B, D2, J; JF vs J), reference scores + trust layer.
#   3. summary tables (code/summary_final.py).
# No YOLOv8 runs meanwhile (fold 4 paused by the user until all of this is done).  Restartable: skips finished steps.
set -e
LANE=final
source /Users/uyen/Downloads/BONEVQA/yolo_pilot/code/folds_lib.sh
FOLDS=(${=FOLDS:-0 1 2 3})
L=0.02
SW=$VGDIR/runs/crossfit_yolo_joint_sweep
NAME=JF_v4cf_yolojoint_btxrd_r$L
V4TRAIN="scripts\.(train_arm|train_joint)"
SEL=$VGDIR/runs/joint/final_btxrd_level.json
BTLEVELS=(0.02 0.05 0.10 0.15 0.30)
evalarm() {  # fold, name, then eval_arm arguments
  local f=$1 name=$2; shift 2
  for role in test calibration; do
    if [ ! -f $VGDIR/runs/yolo_ablation/fold_$f/${name}_$role.json ]; then
      stamp "eval $name fold $f $role"
      v4 -m scripts.eval_arm --name $name --fold $f --role $role "$@" > $LOG/eval_${name}_f${f}_$role.log 2>&1
    fi
  done
}
evaljoint() {  # fold, name, tag, FracAtlas proposals, BTXRD proposals, roles
  local f=$1 name=$2 tag=$3 fap=$4 btp=$5; shift 5
  for role in "$@"; do
    if [ ! -f $VGDIR/runs/joint/fold_$f/${name}_btxrd_$role.json ]; then
      stamp "eval $name fold $f BTXRD $role"
      v4 -m scripts.eval_joint --name $name --tag $tag --fold $f --role $role --proposals-file $fap \
        --btxrd-vqa $BTQ --btxrd-proposals $btp > $LOG/evaljoint_${name}_f${f}_$role.log 2>&1
    fi
  done
}
trustchain() {  # fold, name, tag, FracAtlas proposals, BTXRD proposals
  local f=$1 name=$2 tag=$3 fap=$4 btp=$5
  for role in model_val test external; do
    if [ ! -f $VGDIR/runs/joint/fold_$f/ref_${name}_$role.json ]; then
      stamp "reference scores $name fold $f $role"
      v4 -m scripts.reference_score --name $name --tag $tag --fold $f --role $role --proposals-file $fap \
        --btxrd-vqa $BTQ --btxrd-proposals $btp > $LOG/ref_${name}_f${f}_$role.log 2>&1
    fi
  done
  if [ ! -f $VGDIR/runs/joint/fold_$f/trust_${name}.json ]; then
    stamp "trust layer $name fold $f"
    v4 -m scripts.trust_fit $name --fold $f > $LOG/trust_${name}_f$f.log 2>&1
  fi
}
trainjf() {  # fold
  local f=$1 tag=final_fold$1/V4-CF_yolojoint_btxrd_r$L
  if ! trained $tag $f 42; then
    while pgrep -f "$V4TRAIN" > /dev/null; do sleep 60; done
    if ! trained $tag $f 42; then
      diskcheck; stamp "train $NAME fold $f"
      v4 -m scripts.train_joint --arm V4-CF --fold $f --seed 42 --tag $tag --device cpu --threads 6 \
        --proposals-file $SW/r$L/fold_$f/proposals_fold$f.json --btxrd-vqa $BTQ \
        --btxrd-proposals $SW/r$L/btxrd_fold$f/proposals_btxrd_insample.json > $LOG/train_final_f$f.log 2>&1
    fi
  fi
  trained $tag $f 42 || { stamp "JF fold $f did not complete"; exit 4; }
}

{
stamp "final configuration $NAME on folds $FOLDS (scope freeze): start"
# ---------------------------------------------------------------- 1. BTXRD level on fold 0 model_val (inference only)
trainjf 0
if [ ! -f $SEL ]; then
  for l in $BTLEVELS; do
    evaljoint 0 JFsel_bt${l} final_fold0/V4-CF_yolojoint_btxrd_r$L $SW/r$L/fold_0/proposals_fold0.json \
      $SW/r$l/btxrd_fold0/proposals_btxrd_insample.json model_val
  done
  python3 -I - $SEL ${BTLEVELS[@]} <<'EOF'
import json, sys
sel, levels = sys.argv[1], sys.argv[2:]
J = "/Users/uyen/Downloads/BONEVQA/v4_grounded_next/runs/joint/fold_0"
cand = {}
for l in levels:
    m = json.load(open(f"{J}/JFsel_bt{l}_btxrd_model_val.json"))["metrics"]
    cand[l] = {"tumor_auroc": m["tumor"]["auroc"], "tumor_auprc": m["tumor"]["auprc"],
               "malignancy_auroc": m["malignancy"]["auroc"], "diagnosis_accuracy": m["diagnosis"]["accuracy"]}
# rule declared before looking: highest tumour AUROC on BTXRD model_val of fold 0; ties (< 0.001) -> the stricter level
best = max(levels, key=lambda l: (round(cand[l]["tumor_auroc"], 3), -float(l)))
json.dump({"rule": "max tumour AUROC on BTXRD model_val, fold 0, JF trained with BTXRD level 0.02; inference only; "
                   "ties -> stricter level", "chosen_level": best, "candidates": cand, "test_or_external_used": False},
          open(sel, "x"), indent=2)
print("chosen BTXRD level", best, cand)
EOF
fi
BL=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['chosen_level'])" $SEL)
stamp "BTXRD inference level locked: $BL (fold 0 model_val)"

# ---------------------------------------------------------------- 2. folds one at a time
for f in $FOLDS; do
  P=$SW/r$L/fold_$f/proposals_fold$f.json
  BTP=$SW/r$BL/btxrd_fold$f/proposals_btxrd_insample.json
  tag=final_fold$f/V4-CF_yolojoint_btxrd_r$L
  for x in $P $BTP; do [ -f $x ] || { stamp "missing $x"; exit 3; }; done
  trainjf $f
  evalarm $f $NAME --tag $tag --proposals-file $P
  evaljoint $f $NAME $tag $P $BTP model_val test external
  D=$VGDIR/runs/yolo_ablation/fold_$f
  if [ ! -f $D/compare_test_final.json ]; then
    stamp "compare fold $f"
    v4 -m scripts.compare_arms --fold $f --role test --out compare_test_final A_v4_sealed_frcnn B_v4cf_frcnn D2_v4cf_yolojoint \
      J_v4cf_yolojoint_btxrd $NAME > $LOG/compare_test_final_f$f.log 2>&1 || stamp "FracAtlas comparison fold $f failed"
    v4 -m scripts.compare_arms --fold $f --role test --out compare_test_final_vsJ J_v4cf_yolojoint_btxrd $NAME \
      > $LOG/compare_test_final_vsJ_f$f.log 2>&1 || stamp "FracAtlas comparison vs J fold $f failed"
  fi
  for role in test external; do
    if [ ! -f $VGDIR/runs/joint/fold_$f/compare_btxrd_${role}_JF_vs_J.json ]; then
      v4 -m scripts.compare_btxrd --fold $f --role $role --out compare_btxrd_${role}_JF_vs_J --btxrd-vqa $BTQ \
        J_v4cf_yolojoint_btxrd $NAME > $LOG/compare_btxrd_final_${role}_f$f.log 2>&1 || stamp "BTXRD comparison fold $f $role failed"
    fi
  done
  stamp "final fold $f results ready"
  trustchain $f $NAME $tag $P $BTP
  stamp "final fold $f done (trust layer)"
done
stamp "final configuration done"

# ---------------------------------------------------------------- 3. summary tables
cd $PILOT
python3 -I code/summary_final.py > $LOG/summary_final.log 2>&1 && stamp "summary tables written (runs/summary_final/)" \
  || stamp "summary tables failed (see logs/folds/summary_final.log)"
} >> $QLOG
