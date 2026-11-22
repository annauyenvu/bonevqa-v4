#!/bin/zsh
# Threshold sweep of the YOLOv8 proposals on fold 0 (approved 2026-11-19: replaces fold 4 in the thesis schedule).
#   1. once the GPU lane has written the fold 3 proposals, stop it (fold 4 detectors wait): V4 then has the power
#      adapter to itself (the 68 W adapter cannot feed V4 + YOLOv8).
#   2. once the CPU lane has finished fold 3, stop it (it only waits for fold 4) and train D2 on fold 0 with the
#      proposals of every sweep level (code/sweep_proposals.py), evaluate on test, compare.
#   3. restart both lanes for fold 4 (IEEE).
# Same D2 recipe as fold 0 (train_arm V4-CF, seed 42, CPU); only the proposal file changes.  Level chosen on model_val.
set -e
LANE=sweep
source /Users/uyen/Downloads/BONEVQA/yolo_pilot/code/folds_lib.sh
LEVELS=(${=LEVELS:-0.02 0.15 max})
SW=$VGDIR/runs/crossfit_yolo_joint_sweep
V4PAT="scripts\.(train_arm|train_joint|eval_arm|eval_joint|reference_score|trust_fit|compare_arms|compare_btxrd)"
{
stamp "sweep lane start (levels: $LEVELS)"

# ---- 1. hold the GPU lane after fold 3
while [ ! -f $LANES/props_f3.done ]; do sleep 20; done
sleep 30   # let the GPU lane write its "fold 3 done" stamp
pkill -f "code/lane_gp[u].sh" || true
sleep 5
pkill -f "train_yolo\.py --name fa_btxrd_os3_f4" || true
stamp "GPU lane stopped after fold 3 (fold 4 detectors resume after the sweep)"

# ---- 2. wait for the CPU lane to finish fold 3, then hold it
while ! grep -q "\[cpu\] fold 3 done" $QLOG; do sleep 120; done
while pgrep -f "$V4PAT" > /dev/null; do sleep 30; done
pkill -f "code/lane_cp[u].sh" || true
stamp "CPU lane stopped after fold 3"

names=()
for l in $LEVELS; do
  P=$SW/r$l/fold_0/proposals_fold0.json
  tag=yolo_fold0_sweep/V4-CF_yolojoint_r$l
  name=D2_sweep_r$l
  names+=($name)
  [ -f $P ] || { stamp "missing $P"; exit 3; }
  if ! trained $tag 0 42; then
    diskcheck; stamp "train D2 sweep level $l fold 0"
    v4 -m scripts.train_arm --arm V4-CF --fold 0 --seed 42 --device cpu --threads 6 --tag $tag --proposals-file $P \
      > $LOG/train_sweep_r$l.log 2>&1
  fi
  if [ ! -f $VGDIR/runs/yolo_ablation/fold_0/${name}_test.json ]; then
    stamp "eval $name fold 0 test"
    v4 -m scripts.eval_arm --name $name --tag $tag --fold 0 --role test --proposals-file $P > $LOG/eval_${name}_test.log 2>&1
  fi
done
stamp "compare sweep fold 0"
v4 -m scripts.compare_arms --fold 0 --role test --out compare_test_sweep B_v4cf_frcnn D2_v4cf_yolojoint ${names[@]} \
  > $LOG/compare_test_sweep.log 2>&1 || stamp "sweep comparison failed"
stamp "sweep done"

# ---- 3. fold 4 (IEEE): both lanes again, they skip every finished step
cd $PILOT
DISKMIN=5 FOLDS=4 nohup caffeinate -is ./code/lane_gpu.sh > logs/lane_gpu.out 2>&1 &
DISKMIN=5 FOLDS=4 nohup caffeinate -is ./code/lane_cpu.sh > logs/lane_cpu.out 2>&1 &
stamp "lanes restarted for fold 4"
} >> $QLOG
