#!/bin/zsh
# After lane_sweep.sh: placebo / oracle-location evaluations of every sweep level (inference only, CPU, test role),
# their comparisons, then code/sweep_analysis.py report.  Asked 2026-11-19 (when is YOLOv8 best, box correctness,
# grounding).  Runs while the CPU lane waits for the fold 4 proposals; never next to a V4 training.
set -e
LANE=analysis
source /Users/uyen/Downloads/BONEVQA/yolo_pilot/code/folds_lib.sh
LEVELS=(0.02 0.15 0.2840909090909091 max)
SW=$VGDIR/runs/crossfit_yolo_joint_sweep
ABL=$VGDIR/runs/yolo_ablation/fold_0
idle() { while pgrep -f "scripts\.(train_arm|train_joint)" > /dev/null; do sleep 60; done }
{
while ! grep -q "\[sweep\] sweep done" $QLOG; do sleep 120; done
stamp "sweep analysis start"
for l in $LEVELS; do
  if [[ $l == 0.284* ]]; then name=D2_v4cf_yolojoint; tag=yolo_fold0/V4-CF_yolojoint
  else name=D2_sweep_r$l; tag=yolo_fold0_sweep/V4-CF_yolojoint_r$l; fi
  [ -f $ABL/${name}_test.json ] || { stamp "missing ${name}_test.json, level $l skipped"; continue; }
  cd $PILOT; $YPY -I code/sweep_analysis.py make $l > $LOG/sweep_make_r$l.log 2>&1
  for k in placebo oracleloc; do
    if [ ! -f $ABL/${name}_${k}_test.json ]; then
      idle; stamp "eval ${name}_$k fold 0 test"
      v4 -m scripts.eval_arm --name ${name}_$k --tag $tag --fold 0 --role test \
        --proposals-file $SW/r$l/fold_0/proposals_fold0_$k.json > $LOG/eval_${name}_${k}_test.log 2>&1
    fi
  done
  v4 -m scripts.compare_arms --fold 0 --role test --out compare_test_loc_r$l $name ${name}_placebo ${name}_oracleloc \
    > $LOG/compare_test_loc_r$l.log 2>&1 || stamp "location comparison $l failed"
done
cd $PILOT; $YPY -I code/sweep_analysis.py report > $LOG/sweep_report.log 2>&1 || stamp "sweep report failed"
stamp "sweep analysis done (runs/sweep_analysis/fold0.md)"
} >> $QLOG
