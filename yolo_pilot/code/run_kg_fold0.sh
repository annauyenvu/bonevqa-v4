#!/bin/zsh
# Fold-0 knowledge-graph runs, queued behind the joint run J (never two trainings at once).
# Both retrain J's recipe (V4-CF, joint YOLOv8 proposals, FracAtlas + BTXRD) with the KG latent prompt:
#   K    bank = the 47 knowledge cards of kg_rgo/out/v2/bone_kg_v2.json
#   K0   bank = empty: the same new parameters, no knowledge (capacity control)
# J itself is the "no module" reference.  The gate kg.gamma gets MULT x the learning rate in both runs.
# Then, per run: FracAtlas fracture endpoint on test; BTXRD endpoints on val, test, external.
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
MULT=10
stamp() { date "+%F %T $1" }
v4() { ( source $VGDIR/env.sh > /dev/null; cd $VGDIR; $PY "$@" ) }

stamp "waiting for the joint run J"
while pgrep -f "code/run_joint_fold[0]\.sh" > /dev/null; do sleep 120; done   # the chain itself, not shells that only name its log
if ! grep -q " done$" $LOG/run_joint_fold0.log; then
  stamp "the joint chain did not finish: not starting"; exit 1
fi

# The gate learning-rate multiplier has not been run before: two updates first, nothing saved.
SMK=smoke_kg/V4-CF_kg_gate_$(date +%s)
stamp "smoke test of the gate multiplier ($SMK)"
v4 -m scripts.with_kg train_joint --kg-bank kg --kg-gamma-lr-mult $MULT -- --arm V4-CF --tag $SMK --device cpu --threads 6 \
  --proposals-file $CFJ/proposals_fold0.json --btxrd-vqa $BTQ --btxrd-proposals $BTP \
  --max-updates 2 --eval-limit 4 --no-save > $LOG/kg_smoke_gate.log 2>&1

for bank name suffix in kg K_v4cf_kg kg  empty K0_v4cf_kgempty kgempty; do
  TAG=joint_fold0/V4-CF_yolojoint_btxrd_$suffix
  if [ ! -f $VGDIR/runs/arms/$TAG/fold_0/seed_42/completed.json ]; then
    stamp "train $TAG"
    v4 -m scripts.with_kg train_joint --kg-bank $bank --kg-gamma-lr-mult $MULT -- --arm V4-CF --tag $TAG --device cpu --threads 6 \
      --proposals-file $CFJ/proposals_fold0.json --btxrd-vqa $BTQ --btxrd-proposals $BTP > $LOG/train_kg_${suffix}_fold0.log 2>&1
  fi
  if [ ! -f $ABL/${name}_test.json ]; then
    stamp "eval $name on FracAtlas test"
    v4 -m scripts.with_kg eval_arm -- --name $name --tag $TAG --role test \
      --proposals-file $CFJ/proposals_fold0.json > $LOG/eval_${name}_fracatlas.log 2>&1
  fi
  for role in model_val test external; do
    if [ ! -f $JNT/${name}_btxrd_${role}.json ]; then
      stamp "eval $name on BTXRD $role"
      v4 -m scripts.with_kg eval_joint -- --name $name --tag $TAG --role $role --proposals-file $CFJ/proposals_fold0.json \
        --btxrd-vqa $BTQ --btxrd-proposals $BTP > $LOG/eval_joint_${name}_${role}.log 2>&1
    fi
  done
done

stamp "compare FracAtlas test"
v4 -m scripts.compare_arms --role test --out compare_test_kg J_v4cf_yolojoint_btxrd K_v4cf_kg K0_v4cf_kgempty D2_v4cf_yolojoint \
  > $LOG/compare_test_kg.log 2>&1 || stamp "FracAtlas comparison failed (see compare_test_kg.log)"
for role in model_val test external; do
  stamp "compare BTXRD $role"
  v4 -m scripts.compare_btxrd --role $role --out compare_btxrd_${role}_kg --btxrd-vqa $BTQ \
    J_v4cf_yolojoint_btxrd K_v4cf_kg K0_v4cf_kgempty > $LOG/compare_btxrd_${role}_kg.log 2>&1 \
    || stamp "BTXRD comparison failed for $role (see the log)"
done
stamp "done"
