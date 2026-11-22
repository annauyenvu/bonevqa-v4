#!/bin/zsh
# After the KG chain: does reference information (age, sex, site, view, region, hardware) help J, K and K0?
# Inference only.  Per run: fracture predictions on the FracAtlas calibration role, BTXRD reference scores on
# val / test / external (scripts/reference_score.py), then one analysis (scripts/reference_analyze.py).
# Every step is skipped when its output exists.
PILOT=/Users/uyen/Downloads/BONEVQA/yolo_pilot
VGDIR=/Users/uyen/Downloads/BONEVQA/v4_grounded_next
LOG=$PILOT/logs
CFJ=$VGDIR/runs/crossfit_yolo_joint/fold_0
BTP=$VGDIR/runs/crossfit_yolo_joint/btxrd/proposals_btxrd_insample.json
BTQ=$PILOT/datasets/btxrd_vqa
ABL=$VGDIR/runs/yolo_ablation/fold_0
JNT=$VGDIR/runs/joint/fold_0
stamp() { date "+%F %T $1" }
v4() { ( source $VGDIR/env.sh > /dev/null; cd $VGDIR; $PY "$@" ) }

stamp "waiting for the KG chain"
while pgrep -f "code/run_kg_fold[0]\.sh" > /dev/null || pgrep -f "code/run_kg_report_fold[0]\.sh" > /dev/null; do sleep 120; done
names=()
for name tag in J_v4cf_yolojoint_btxrd joint_fold0/V4-CF_yolojoint_btxrd K_v4cf_kg joint_fold0/V4-CF_yolojoint_btxrd_kg \
                K0_v4cf_kgempty joint_fold0/V4-CF_yolojoint_btxrd_kgempty; do
  if [ ! -f $VGDIR/runs/arms/$tag/fold_0/seed_42/completed.json ]; then stamp "$name was not trained: skipped"; continue; fi
  if [ ! -f $ABL/${name}_calibration.json ]; then
    stamp "eval $name on FracAtlas calibration"
    v4 -m scripts.with_kg eval_arm -- --name $name --tag $tag --role calibration \
      --proposals-file $CFJ/proposals_fold0.json > $LOG/eval_${name}_calibration.log 2>&1 || stamp "calibration eval failed for $name"
  fi
  ok=1
  for role in model_val test external; do
    if [ ! -f $JNT/ref_${name}_${role}.json ]; then
      stamp "reference scores $name $role"
      v4 -m scripts.reference_score --name $name --tag $tag --role $role --proposals-file $CFJ/proposals_fold0.json \
        --btxrd-vqa $BTQ --btxrd-proposals $BTP > $LOG/ref_${name}_${role}.log 2>&1 || { stamp "reference scores failed for $name $role"; ok=0 }
    fi
  done
  [ $ok = 1 ] && names+=($name)
done
if [ ${#names} -gt 0 ]; then
  stamp "analysis of ${names[*]}"
  OUT=reference_fold0_$(date +%m%d_%H%M)
  v4 -m scripts.reference_analyze --out $OUT --btxrd-vqa $BTQ ${names[@]} > $LOG/reference_analyze.log 2>&1 \
    && stamp "written $JNT/$OUT.md" || stamp "analysis failed (see reference_analyze.log)"
fi
stamp "done"
