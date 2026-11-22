#!/bin/zsh
# Fold-0 V4 ablation with YOLOv8 proposals.  One job at a time: V4 trains on CPU only
# (MPS gradients are wrong for this model), YOLOv8 trains on MPS.
#   A   sealed V4 checkpoint, Faster R-CNN proposals                     (no training)
#   E   sealed V4 checkpoint, YOLOv8 (FracAtlas) proposals at test only  (no training)
#   E2  sealed V4 checkpoint, YOLOv8 (FracAtlas+BTXRD) proposals at test (no training)
#   C2  V4 retrained, YOLOv8 (FracAtlas+BTXRD) in-sample TRAIN proposals (same design as V4)
#   D2  V4 retrained, YOLOv8 (FracAtlas+BTXRD) cross-fitted TRAIN proposals
#   B   V4 retrained, Faster R-CNN cross-fitted TRAIN proposals, full schedule
#   C   V4 retrained, YOLOv8 (FracAtlas) in-sample TRAIN proposals
#   D   V4 retrained, YOLOv8 (FracAtlas) cross-fitted TRAIN proposals
# The joint detector goes first because it is the better one on model_val (box AP50,
# top-3 recall and image-level AP), the rule fixed before test was read for V4.
# Every step is skipped when its output exists, so the script can be restarted.
# The V4 commands run in a subshell that sources env.sh, so its PYTHONPATH and MPS
# settings never reach the YOLOv8 interpreter.
set -e
PILOT=/Users/uyen/Downloads/BONEVQA/yolo_pilot
VGDIR=/Users/uyen/Downloads/BONEVQA/v4_grounded_next
LOG=$PILOT/logs
CF=$VGDIR/runs/crossfit_yolo/fold_0
CFJ=$VGDIR/runs/crossfit_yolo_joint/fold_0
ABL=$VGDIR/runs/yolo_ablation/fold_0
YPY=$PILOT/venv/bin/python
stamp() { date "+%F %T $1" }
v4() { ( source $VGDIR/env.sh > /dev/null; cd $VGDIR; $PY "$@" ) }

evalarm() {  # name, then eval_arm arguments
  local name=$1; shift
  if [ ! -f $ABL/${name}_test.json ]; then
    stamp "eval $name"
    v4 -m scripts.eval_arm --name $name --role test "$@" > $LOG/eval_$name.log 2>&1
  fi
}
trainarm() {  # tag, arm, then train_arm arguments
  local tag=$1 arm=$2; shift 2
  if [ ! -f $VGDIR/runs/arms/$tag/fold_0/seed_42/completed.json ]; then
    stamp "train $tag"
    v4 -m scripts.train_arm --arm $arm --fold 0 --seed 42 --device cpu --threads 6 --tag $tag "$@" > $LOG/train_${tag:t}.log 2>&1
  fi
}
inner() {  # run-name prefix (cf | cfj), variant (fa | joint), proposals dir
  local prefix=$1 variant=$2 dir=$3
  cd $PILOT
  for k in 0 1 2; do
    if [ ! -f runs/yolo/${prefix}${k}/done ]; then
      if [ -d runs/yolo/${prefix}${k} ]; then mv runs/yolo/${prefix}${k} runs/yolo/${prefix}${k}_incomplete_$(date +%s); fi
      stamp "train inner detector ${prefix}${k}"
      $YPY code/train_yolo.py --name ${prefix}${k} --data datasets/$4${k}.yaml --epochs 60 --patience 20 --device mps > logs/train_${prefix}${k}.log 2>&1
      touch runs/yolo/${prefix}${k}/done
    fi
    if [ ! -f runs/pred/${prefix}${k}_cls0.json ]; then
      stamp "predict inner detector ${prefix}${k}"
      $YPY code/predict_yolo.py --weights runs/yolo/${prefix}${k}/weights/best.pt --out runs/pred/${prefix}${k} --lists datasets/fa_fold0/cf/inner${k}_heldout.txt --dirs datasets/fa_fold0/images/val --device cpu > logs/predict_${prefix}${k}.log 2>&1
    fi
  done
  if [ ! -f $dir/proposals_fold0.json ]; then
    stamp "build cross-fitted proposals ($variant)"
    python3 -I code/build_v4_proposals.py 0 full $variant > logs/build_v4_proposals_full_$variant.log 2>&1
  fi
}

cd $PILOT
if [ ! -f $CFJ/proposals_fold0_insample.json ]; then
  python3 -I code/build_v4_proposals.py 0 insample joint > logs/build_v4_proposals_insample_joint.log 2>&1
fi

evalarm A_v4_sealed_frcnn --sealed-v4
evalarm E_v4_sealed_yolo --sealed-v4 --proposals-file $CF/proposals_fold0_insample.json
evalarm E2_v4_sealed_yolojoint --sealed-v4 --proposals-file $CFJ/proposals_fold0_insample.json

trainarm yolo_fold0/V4_yolojoint_insample V4 --proposals-file $CFJ/proposals_fold0_insample.json
evalarm C2_v4_yolojoint_insample --tag yolo_fold0/V4_yolojoint_insample --proposals-file $CFJ/proposals_fold0_insample.json

inner cfj joint $CFJ data_cfj
trainarm yolo_fold0/V4-CF_yolojoint V4-CF --proposals-file $CFJ/proposals_fold0.json
evalarm D2_v4cf_yolojoint --tag yolo_fold0/V4-CF_yolojoint --proposals-file $CFJ/proposals_fold0.json

trainarm yolo_fold0/V4-CF_frcnn_full V4-CF
evalarm B_v4cf_frcnn --tag yolo_fold0/V4-CF_frcnn_full
stamp "compare (joint detector)"
v4 -m scripts.compare_arms --role test --out compare_test_joint A_v4_sealed_frcnn B_v4cf_frcnn E_v4_sealed_yolo E2_v4_sealed_yolojoint C2_v4_yolojoint_insample D2_v4cf_yolojoint > $LOG/compare_test_joint.log 2>&1

trainarm yolo_fold0/V4_yolo_insample V4 --proposals-file $CF/proposals_fold0_insample.json
evalarm C_v4_yolo_insample --tag yolo_fold0/V4_yolo_insample --proposals-file $CF/proposals_fold0_insample.json
inner cf fa $CF fa_fold0/cf/data_cf
trainarm yolo_fold0/V4-CF_yolo V4-CF --proposals-file $CF/proposals_fold0.json
evalarm D_v4cf_yolo --tag yolo_fold0/V4-CF_yolo --proposals-file $CF/proposals_fold0.json

stamp "compare (all)"
v4 -m scripts.compare_arms --role test --out compare_test_all A_v4_sealed_frcnn B_v4cf_frcnn E_v4_sealed_yolo E2_v4_sealed_yolojoint C_v4_yolo_insample C2_v4_yolojoint_insample D_v4cf_yolo D2_v4cf_yolojoint > $LOG/compare_test_all.log 2>&1
stamp "done"
