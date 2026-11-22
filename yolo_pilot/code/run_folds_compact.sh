#!/bin/zsh
# Compact multi-fold package (approved 2026-11-17: "Gói gọn", "Giữ B").  One job at a time, never two trainings:
#   part 1  fold 0, joint run J with seeds 43 and 44 (same detector and proposals as J seed 42)
#   part 2  folds 1-4: joint YOLOv8 detector (outer + 3 inner cross-fit), proposals, then
#           B  V4-CF, Faster R-CNN cross-fitted proposals (control)
#           D2 V4-CF, joint YOLOv8 cross-fitted proposals
#           J  D2 recipe + BTXRD QA (joint training)
#           A  sealed V4 of the fold (no training, evaluation only)
#   evaluation: FracAtlas test (A, B, D2, J), calibration (J), BTXRD val/test/external (D2, J),
#   reference scores and trust layer (J), paired comparisons per fold.
# YOLOv8 trains on MPS, V4 on CPU only (MPS gradient bug).  Every step is skipped when its output exists,
# so the script can be restarted; a detector folder without its "done" marker is moved aside and retrained.
# Stops when free disk space falls below 5 GB.
set -e
PILOT=/Users/uyen/Downloads/BONEVQA/yolo_pilot
VGDIR=/Users/uyen/Downloads/BONEVQA/v4_grounded_next
LOG=$PILOT/logs/folds
mkdir -p $LOG
YPY=$PILOT/venv/bin/python
BTQ=$PILOT/datasets/btxrd_vqa
FOLDS=(${=FOLDS:-1 2 3 4})
SEEDS=(${=SEEDS:-43 44})
stamp() { date "+%F %T $1" }
v4() { ( source $VGDIR/env.sh > /dev/null; cd $VGDIR; $PY "$@" ) }
diskcheck() {
  local free=$(df -g /Users/uyen | awk 'NR==2 {print $4}')
  if [ "$free" -lt 5 ]; then stamp "STOP: only ${free} GB free"; exit 2; fi
}
trained() { [ -f $VGDIR/runs/arms/$1/fold_$2/seed_$3/completed.json ] }

yolo_train() {  # run name, data yaml (relative to yolo_pilot)
  local name=$1 data=$2
  cd $PILOT
  if [ ! -f runs/yolo/$name/done ]; then
    diskcheck
    if [ -d runs/yolo/$name ]; then mv runs/yolo/$name runs/yolo/${name}_incomplete_$(date +%s); fi
    stamp "train detector $name"
    $YPY code/train_yolo.py --name $name --data $data --epochs 60 --patience 20 --device mps > $LOG/train_$name.log 2>&1
    touch runs/yolo/$name/done
  fi
}
yolo_predict() {  # weights run name, output name, then predict_yolo arguments
  local name=$1 out=$2; shift 2
  cd $PILOT
  if [ ! -f runs/pred/${out}_cls0.json ]; then
    stamp "predict $out"
    $YPY code/predict_yolo.py --weights runs/yolo/$name/weights/best.pt --out runs/pred/$out "$@" --device cpu > $LOG/predict_$out.log 2>&1
  fi
}
evalarm() {  # fold, name, then eval_arm arguments
  local f=$1 name=$2; shift 2
  for role in test calibration; do
    if [ $role = calibration ] && [[ $name != J* ]]; then continue; fi
    if [ ! -f $VGDIR/runs/yolo_ablation/fold_$f/${name}_$role.json ]; then
      stamp "eval $name fold $f $role"
      v4 -m scripts.eval_arm --name $name --fold $f --role $role "$@" > $LOG/eval_${name}_f${f}_$role.log 2>&1
    fi
  done
}
evaljoint() {  # fold, name, tag, FracAtlas proposals, BTXRD proposals
  local f=$1 name=$2 tag=$3 fap=$4 btp=$5
  for role in model_val test external; do
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
trainjoint() {  # fold, seed, tag, FracAtlas proposals, BTXRD proposals
  local f=$1 s=$2 tag=$3 fap=$4 btp=$5
  if ! trained $tag $f $s; then
    diskcheck
    stamp "train $tag fold $f seed $s"
    v4 -m scripts.train_joint --arm V4-CF --fold $f --seed $s --tag $tag --device cpu --threads 6 \
      --proposals-file $fap --btxrd-vqa $BTQ --btxrd-proposals $btp > $LOG/train_${tag:h}_${tag:t}_s$s.log 2>&1
  fi
}

stamp "start (folds: $FOLDS; extra seeds on fold 0: $SEEDS)"
if pgrep -f "scripts\.(train_arm|train_joint)|code/train_yolo\.py" > /dev/null; then
  stamp "a training is already running: not starting"; exit 1
fi

# ------------------------------------------------------------------ part 1: seeds of J on fold 0
CF0=$VGDIR/runs/crossfit_yolo_joint/fold_0/proposals_fold0.json
BT0=$VGDIR/runs/crossfit_yolo_joint/btxrd/proposals_btxrd_insample.json
for s in $SEEDS; do
  tag=joint_fold0_s$s/V4-CF_yolojoint_btxrd
  name=J_s${s}_v4cf_yolojoint_btxrd
  trainjoint 0 $s $tag $CF0 $BT0
  evalarm 0 $name --tag $tag --proposals-file $CF0
  evaljoint 0 $name $tag $CF0 $BT0
  trustchain 0 $name $tag $CF0 $BT0
done
stamp "compare seeds on fold 0"
seednames=()
for s in $SEEDS; do seednames+=(J_s${s}_v4cf_yolojoint_btxrd); done
v4 -m scripts.compare_arms --fold 0 --role test --out compare_test_J_seeds J_v4cf_yolojoint_btxrd ${seednames[@]} \
  > $LOG/compare_test_J_seeds.log 2>&1 || stamp "seed comparison failed"

# ------------------------------------------------------------------ part 2: folds 1-4
for f in $FOLDS; do
  cd $PILOT
  DS=datasets/fa_fold$f
  CFJ=$VGDIR/runs/crossfit_yolo_joint/fold_$f
  BTP=$VGDIR/runs/crossfit_yolo_joint/btxrd_fold$f/proposals_btxrd_insample.json
  if [ ! -f $DS/gt.json ]; then stamp "prepare FracAtlas fold $f for YOLO"; python3 -I code/prep_fracatlas.py $f > $LOG/prep_f$f.log 2>&1; fi
  if [ ! -f $VGDIR/runs/crossfit_yolo/fold_$f/inner_split.json ]; then stamp "inner split fold $f"; v4 -m scripts.yolo_cf_split --fold $f > $LOG/split_f$f.log 2>&1; fi
  cd $PILOT
  if [ ! -d $DS/cf ]; then python3 -I code/make_cf_lists.py $f > $LOG/cflists_f$f.log 2>&1; fi
  if [ ! -f $DS/data_fa_btxrd_os3.yaml ]; then python3 -I code/prep_fold_yaml.py $f > $LOG/yaml_f$f.log 2>&1; fi
  if [ ! -f runs/frcnn/fold${f}_ep2_sealed.json ]; then
    stamp "Faster R-CNN reference boxes fold $f"
    ( source $VGDIR/env.sh > /dev/null; $PY $PILOT/code/frcnn_raw.py $f ) > $LOG/frcnn_raw_f$f.log 2>&1
  fi

  # outer joint detector and its predictions (held-out roles, in-sample TRAIN of both datasets)
  yolo_train fa_btxrd_os3_f$f $DS/data_fa_btxrd_os3.yaml
  yolo_predict fa_btxrd_os3_f$f f${f}_fa_btxrd --dirs $DS/images/val $DS/images/cal $DS/images/test \
    datasets/btxrd_pilot/images/val datasets/btxrd_pilot/images/test datasets/btxrd_pilot/images/external
  yolo_predict fa_btxrd_os3_f$f f${f}_fa_btxrd_train --dirs $DS/images/train
  yolo_predict fa_btxrd_os3_f$f f${f}_fa_btxrd_bttrain --dirs datasets/btxrd_pilot/images/train
  cd $PILOT
  if [ ! -f $CFJ/proposals_fold${f}_insample.json ]; then
    stamp "in-sample proposals fold $f"; python3 -I code/build_v4_proposals.py $f insample joint > $LOG/props_insample_f$f.log 2>&1
  fi
  if [ ! -f $BTP ]; then stamp "BTXRD proposals fold $f"; python3 -I code/build_btxrd_proposals.py $f > $LOG/props_btxrd_f$f.log 2>&1; fi

  # three inner joint detectors for the cross-fitted TRAIN proposals
  for i in 0 1 2; do
    yolo_train cfj${i}_f$f $DS/cf/data_cfj$i.yaml
    yolo_predict cfj${i}_f$f f${f}_cfj$i --lists $DS/cf/inner${i}_heldout.txt --dirs $DS/images/val
  done
  cd $PILOT
  if [ ! -f $CFJ/proposals_fold$f.json ]; then
    stamp "cross-fitted proposals fold $f"; python3 -I code/build_v4_proposals.py $f full joint > $LOG/props_full_f$f.log 2>&1
  fi

  # V4 runs
  if ! trained yolo_fold$f/V4-CF_frcnn_full $f 42; then
    diskcheck; stamp "train B fold $f"
    v4 -m scripts.train_arm --arm V4-CF --fold $f --seed 42 --device cpu --threads 6 --tag yolo_fold$f/V4-CF_frcnn_full \
      > $LOG/train_B_f$f.log 2>&1
  fi
  if ! trained yolo_fold$f/V4-CF_yolojoint $f 42; then
    diskcheck; stamp "train D2 fold $f"
    v4 -m scripts.train_arm --arm V4-CF --fold $f --seed 42 --device cpu --threads 6 --tag yolo_fold$f/V4-CF_yolojoint \
      --proposals-file $CFJ/proposals_fold$f.json > $LOG/train_D2_f$f.log 2>&1
  fi
  trainjoint $f 42 joint_fold$f/V4-CF_yolojoint_btxrd $CFJ/proposals_fold$f.json $BTP

  evalarm $f A_v4_sealed_frcnn --sealed-v4
  evalarm $f B_v4cf_frcnn --tag yolo_fold$f/V4-CF_frcnn_full
  evalarm $f D2_v4cf_yolojoint --tag yolo_fold$f/V4-CF_yolojoint --proposals-file $CFJ/proposals_fold$f.json
  evalarm $f J_v4cf_yolojoint_btxrd --tag joint_fold$f/V4-CF_yolojoint_btxrd --proposals-file $CFJ/proposals_fold$f.json
  evaljoint $f D2_v4cf_yolojoint yolo_fold$f/V4-CF_yolojoint $CFJ/proposals_fold$f.json $BTP
  evaljoint $f J_v4cf_yolojoint_btxrd joint_fold$f/V4-CF_yolojoint_btxrd $CFJ/proposals_fold$f.json $BTP
  trustchain $f J_v4cf_yolojoint_btxrd joint_fold$f/V4-CF_yolojoint_btxrd $CFJ/proposals_fold$f.json $BTP

  stamp "compare fold $f"
  v4 -m scripts.compare_arms --fold $f --role test --out compare_test_vsB B_v4cf_frcnn A_v4_sealed_frcnn D2_v4cf_yolojoint J_v4cf_yolojoint_btxrd \
    > $LOG/compare_test_f$f.log 2>&1 || stamp "FracAtlas comparison fold $f failed"
  for role in test external; do
    v4 -m scripts.compare_btxrd --fold $f --role $role --out compare_btxrd_${role}_J_vs_D2 --btxrd-vqa $BTQ \
      D2_v4cf_yolojoint J_v4cf_yolojoint_btxrd > $LOG/compare_btxrd_${role}_f$f.log 2>&1 || stamp "BTXRD comparison fold $f $role failed"
  done
  stamp "fold $f done"
done
stamp "done"
