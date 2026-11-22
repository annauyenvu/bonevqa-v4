#!/bin/zsh
# After the KG run K has been evaluated: gate value, retrieval weights, open / reasoning answers of K vs J.
# Inference only, 2 threads, low priority: it may overlap with the K0 training of run_kg_fold0.sh.
PILOT=/Users/uyen/Downloads/BONEVQA/yolo_pilot
VGDIR=/Users/uyen/Downloads/BONEVQA/v4_grounded_next
JNT=$VGDIR/runs/joint/fold_0
stamp() { date "+%F %T $1" }
stamp "waiting for the evaluation of K"
until [ -f $JNT/K_v4cf_kg_btxrd_external.json ] || ! pgrep -f "code/run_kg_fold[0]\.sh" > /dev/null; do sleep 120; done
if [ ! -f $JNT/K_v4cf_kg_btxrd_external.json ]; then stamp "K was not evaluated: not starting"; exit 1; fi
stamp "kg_report K"
( source $VGDIR/env.sh > /dev/null; cd $VGDIR; export OMP_NUM_THREADS=2
  nice -n 15 $PY -m scripts.kg_report --tag joint_fold0/V4-CF_yolojoint_btxrd_kg --reference-tag joint_fold0/V4-CF_yolojoint_btxrd \
    --out kg_report_K --proposals-file runs/crossfit_yolo_joint/fold_0/proposals_fold0.json \
    --btxrd-vqa ../yolo_pilot/datasets/btxrd_vqa --btxrd-proposals runs/crossfit_yolo_joint/btxrd/proposals_btxrd_insample.json ) > $PILOT/logs/kg_report_K.log 2>&1 \
  && stamp "done" || stamp "kg_report failed (see kg_report_K.log)"
