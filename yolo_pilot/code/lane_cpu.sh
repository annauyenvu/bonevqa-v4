#!/bin/zsh
# CPU lane: every V4 training and evaluation (CPU only, one at a time).
#   1. J seeds 43, 44 on fold 0 (waits for the seed 44 training already running), seed comparison
#   2. B of every fold (needs only the fold preparation of the GPU lane)
#   3. per fold, once the GPU lane wrote its proposals: D2, J, evaluations, comparisons, then reference scores + trust layer
# Restartable: every step is skipped when its output exists.
set -e
LANE=cpu
source /Users/uyen/Downloads/BONEVQA/yolo_pilot/code/folds_lib.sh
FOLDS=(${=FOLDS:-1 2 3 4})
SEEDS=(${=SEEDS:-43 44})

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
trainarm() {  # fold, tag, then extra train_arm arguments
  local f=$1 tag=$2 short=$3; shift 3
  if ! trained $tag $f 42; then
    diskcheck; stamp "train $short fold $f"
    v4 -m scripts.train_arm --arm V4-CF --fold $f --seed 42 --device cpu --threads 6 --tag $tag "$@" > $LOG/train_${short}_f$f.log 2>&1
  fi
}

{
stamp "CPU lane start (folds: $FOLDS; extra seeds on fold 0: $SEEDS)"
# a V4 training started by the earlier single queue may still be running: let it finish, never start a second one
if pgrep -f "scripts\.(train_arm|train_joint)" > /dev/null; then
  stamp "waiting for the V4 training already running"
  while pgrep -f "scripts\.(train_arm|train_joint)" > /dev/null; do sleep 120; done
fi

# ------------------------------------------------------------------ 1. seeds of J on fold 0
CF0=$VGDIR/runs/crossfit_yolo_joint/fold_0/proposals_fold0.json
BT0=$VGDIR/runs/crossfit_yolo_joint/btxrd/proposals_btxrd_insample.json
for s in $SEEDS; do
  tag=joint_fold0_s$s/V4-CF_yolojoint_btxrd
  name=J_s${s}_v4cf_yolojoint_btxrd
  trainjoint 0 $s $tag $CF0 $BT0
  evalarm 0 $name --tag $tag --proposals-file $CF0
  evaljoint 0 $name $tag $CF0 $BT0
done
seednames=()
for s in $SEEDS; do seednames+=(J_s${s}_v4cf_yolojoint_btxrd); done
if [ ! -f $VGDIR/runs/yolo_ablation/fold_0/compare_test_J_seeds.json ]; then
  stamp "compare seeds on fold 0"
  v4 -m scripts.compare_arms --fold 0 --role test --out compare_test_J_seeds J_v4cf_yolojoint_btxrd ${seednames[@]} \
    > $LOG/compare_test_J_seeds.log 2>&1 || stamp "seed comparison failed"
fi

# ------------------------------------------------------------------ 2. B of every fold
for f in $FOLDS; do
  waitfor $LANES/prep_f$f.done
  trainarm $f yolo_fold$f/V4-CF_frcnn_full B
done

# reference scores and trust layer of the extra seeds (while the GPU lane still trains detectors)
for s in $SEEDS; do
  trustchain 0 J_s${s}_v4cf_yolojoint_btxrd joint_fold0_s$s/V4-CF_yolojoint_btxrd $CF0 $BT0
done

# ------------------------------------------------------------------ 3. D2, J, evaluations per fold
for f in $FOLDS; do
  CFJ=$(cfj $f); BTP=$(btp $f); P=$CFJ/proposals_fold$f.json
  waitfor $LANES/props_f$f.done
  trainarm $f yolo_fold$f/V4-CF_yolojoint D2 --proposals-file $P
  trainjoint $f 42 joint_fold$f/V4-CF_yolojoint_btxrd $P $BTP
  evalarm $f A_v4_sealed_frcnn --sealed-v4
  evalarm $f B_v4cf_frcnn --tag yolo_fold$f/V4-CF_frcnn_full
  evalarm $f D2_v4cf_yolojoint --tag yolo_fold$f/V4-CF_yolojoint --proposals-file $P
  evalarm $f J_v4cf_yolojoint_btxrd --tag joint_fold$f/V4-CF_yolojoint_btxrd --proposals-file $P
  evaljoint $f D2_v4cf_yolojoint yolo_fold$f/V4-CF_yolojoint $P $BTP
  evaljoint $f J_v4cf_yolojoint_btxrd joint_fold$f/V4-CF_yolojoint_btxrd $P $BTP
  stamp "compare fold $f"
  v4 -m scripts.compare_arms --fold $f --role test --out compare_test_vsB B_v4cf_frcnn A_v4_sealed_frcnn D2_v4cf_yolojoint J_v4cf_yolojoint_btxrd \
    > $LOG/compare_test_f$f.log 2>&1 || stamp "FracAtlas comparison fold $f failed"
  for role in test external; do
    v4 -m scripts.compare_btxrd --fold $f --role $role --out compare_btxrd_${role}_J_vs_D2 --btxrd-vqa $BTQ \
      D2_v4cf_yolojoint J_v4cf_yolojoint_btxrd > $LOG/compare_btxrd_${role}_f$f.log 2>&1 || stamp "BTXRD comparison fold $f $role failed"
  done
  stamp "fold $f results ready"
  trustchain $f J_v4cf_yolojoint_btxrd joint_fold$f/V4-CF_yolojoint_btxrd $P $BTP
  stamp "fold $f done"
done
stamp "CPU lane done"
} >> $QLOG
