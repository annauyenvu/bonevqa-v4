#!/bin/bash
# Kiểm tra bổ sung cho luận văn trên fold 0 (checkpoint của demo). Có thể chạy lại, điều kiện đã xong được bỏ qua.
ROOT=/Users/uyen/Downloads/BONEVQA; R=$ROOT/refactor_v2; cd "$R" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$ROOT/.evaluation_runtime/hf_cache BONEVQA_DATA=$ROOT/.evaluation_runtime/data PYTHONPATH=$R:$ROOT/.evaluation_runtime/deps
PY=/Users/uyen/Downloads/BoneMedVQA/.venv/bin/python; L=reports/thesis_v4_checks_v1/logs
run(){ $PY -m experiments.thesis_v4_checks_v1 --fold 0 --arm $1 --conds $2 --threads 2 > $L/$3.log 2>&1; }
run V4 reference,delta_off,qcond_off,noise_crops,placebo_boxes w1 &
run V4 iou100,iou050,iou025,iou000,annotation_leak w2 &
run V4 blank,shuffle,down512,down256 w3 &
run V4 jpeg30,blur2,noise10,hflip w4 &
( run V4 rot90,crop80,invert w5; run B reference,blank,shuffle,down512,jpeg30,hflip,rot90 w5b ) &
( for a in V4 B; do for c in blank shuffle; do $PY -m experiments.thesis_v4_qa_sample_v1 $a $c > $L/qa_${a}_$c.log 2>&1; done; done ) &
wait
