#!/bin/zsh
# Level 0.02 of the proposal-threshold sweep on folds 1-3 (approved 2026-11-20 12:4x: "kiểm tra mức 0,02 trên fold 1–3").
# Fold 0 chose 0.02 on model_val (test AUPRC 0.852 vs 0.784 at the FRCNN-matched rate); this checks it on other folds.
# Same D2 recipe (train_arm V4-CF, seed 42, CPU), only the proposal file changes: code/sweep_proposals.py <f> 0.02
# -> v4_grounded_next/runs/crossfit_yolo_joint_sweep/r0.02/fold_<f>/proposals_fold<f>.json.
# Never next to another V4 training (it waits).  User 20/11 13:2x chose this before the fold 4 V4 arms: the CPU lane
# was stopped while it waited for props_f4 and is restarted at the end (FOLDS=4).  Restartable: skips finished steps.
# YOLOv8 keeps running on MPS (no reference_score here: that one blew up swap next to YOLOv8).
set -e
LANE=sweepf
source /Users/uyen/Downloads/BONEVQA/yolo_pilot/code/folds_lib.sh
FOLDS=(${=FOLDS:-1 2 3})
L=0.02
SW=$VGDIR/runs/crossfit_yolo_joint_sweep/r$L
V4TRAIN="scripts\.(train_arm|train_joint)"
{
stamp "level $L on folds $FOLDS: start"
for f in $FOLDS; do
  P=$SW/fold_$f/proposals_fold$f.json
  tag=yolo_fold${f}_sweep/V4-CF_yolojoint_r$L
  name=D2_sweep_r$L
  [ -f $P ] || { stamp "missing $P"; exit 3; }
  if ! trained $tag $f 42; then
    while pgrep -f "$V4TRAIN" > /dev/null; do sleep 60; done
    diskcheck; stamp "train D2 level $L fold $f"
    v4 -m scripts.train_arm --arm V4-CF --fold $f --seed 42 --device cpu --threads 6 --tag $tag --proposals-file $P \
      > $LOG/train_sweep_r${L}_f$f.log 2>&1
  fi
  if [ ! -f $VGDIR/runs/yolo_ablation/fold_$f/${name}_test.json ]; then
    stamp "eval $name fold $f test"
    v4 -m scripts.eval_arm --name $name --tag $tag --fold $f --role test --proposals-file $P > $LOG/eval_${name}_f${f}_test.log 2>&1
  fi
  if [ ! -f $VGDIR/runs/yolo_ablation/fold_$f/compare_test_sweep$L.json ]; then
    stamp "compare fold $f"
    v4 -m scripts.compare_arms --fold $f --role test --out compare_test_sweep$L B_v4cf_frcnn D2_v4cf_yolojoint $name \
      > $LOG/compare_test_sweep${L}_f$f.log 2>&1 || stamp "comparison fold $f failed"
  fi
  stamp "level $L fold $f done"
done
stamp "level $L on folds $FOLDS: done"
# user 20/11 13:2x "Ưu tiên mức 0,02": the CPU lane was stopped while it waited for props_f4; restart it for fold 4
cd $PILOT
if ! pgrep -f "code/lane_cp[u].sh" > /dev/null; then
  DISKMIN=5 FOLDS=4 nohup caffeinate -is ./code/lane_cpu.sh > logs/lane_cpu.out 2>&1 &
  stamp "CPU lane restarted for fold 4"
fi
} >> $QLOG
