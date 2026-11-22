#!/bin/zsh
# Fold-0 joint FracAtlas + BTXRD run, queued behind the V4 ablation chain (never two trainings at once).
#   J   V4-CF retrained on FracAtlas + BTXRD: FracAtlas proposals = joint YOLOv8, cross-fitted (as D2);
#       BTXRD proposals = lesion class of the same detector, in-sample on TRAIN (no cross-fit exists yet).
# Then:
#   FracAtlas fracture endpoint of J on test            -> compare with D2 (same detector and proposals, no BTXRD)
#   BTXRD endpoints of J and of D2 on val, test, external (D2 = a checkpoint that never saw BTXRD)
# Every step is skipped when its output exists, so the script can be restarted.
set -e
PILOT=/Users/uyen/Downloads/BONEVQA/yolo_pilot
VGDIR=/Users/uyen/Downloads/BONEVQA/v4_grounded_next
LOG=$PILOT/logs
CFJ=$VGDIR/runs/crossfit_yolo_joint/fold_0
BTP=$VGDIR/runs/crossfit_yolo_joint/btxrd/proposals_btxrd_insample.json
BTQ=$PILOT/datasets/btxrd_vqa
ABL=$VGDIR/runs/yolo_ablation/fold_0
JNT=$VGDIR/runs/joint/fold_0
TAG=joint_fold0/V4-CF_yolojoint_btxrd
stamp() { date "+%F %T $1" }
v4() { ( source $VGDIR/env.sh > /dev/null; cd $VGDIR; $PY "$@" ) }

stamp "waiting for the V4 ablation chain"
while pgrep -f "run_v4_ablation_fold[0]" > /dev/null; do sleep 120; done
if [ ! -f $ABL/compare_test_all.json ]; then
  stamp "ablation chain stopped without compare_test_all.json: not starting"; exit 1
fi

if [ ! -f $VGDIR/runs/arms/$TAG/fold_0/seed_42/completed.json ]; then
  stamp "train $TAG"
  v4 -m scripts.train_joint --arm V4-CF --tag $TAG --device cpu --threads 6 \
    --proposals-file $CFJ/proposals_fold0.json --btxrd-vqa $BTQ --btxrd-proposals $BTP > $LOG/train_joint_fold0.log 2>&1
fi
if [ ! -f $ABL/J_v4cf_yolojoint_btxrd_test.json ]; then
  stamp "eval J on FracAtlas test"
  v4 -m scripts.eval_arm --name J_v4cf_yolojoint_btxrd --tag $TAG --role test \
    --proposals-file $CFJ/proposals_fold0.json > $LOG/eval_J_fracatlas.log 2>&1
fi
for name tag in J_v4cf_yolojoint_btxrd $TAG D2_v4cf_yolojoint yolo_fold0/V4-CF_yolojoint; do
  for role in model_val test external; do
    if [ ! -f $JNT/${name}_btxrd_${role}.json ]; then
      stamp "eval $name on BTXRD $role"
      v4 -m scripts.eval_joint --name $name --tag $tag --role $role --proposals-file $CFJ/proposals_fold0.json \
        --btxrd-vqa $BTQ --btxrd-proposals $BTP > $LOG/eval_joint_${name}_${role}.log 2>&1
    fi
  done
done
stamp "compare FracAtlas test"
v4 -m scripts.compare_arms --role test --out compare_test_joint_training D2_v4cf_yolojoint J_v4cf_yolojoint_btxrd B_v4cf_frcnn \
  > $LOG/compare_test_joint_training.log 2>&1
stamp "done"
