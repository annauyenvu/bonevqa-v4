#!/bin/bash
# Bổ sung các fold còn thiếu của kiểm tra luận văn để mọi điều kiện đủ 5 fold. Có thể chạy lại, phần đã xong được bỏ qua. Chỉ dùng CPU.
ROOT=/Users/uyen/Downloads/BONEVQA; R=$ROOT/refactor_v2; cd "$R" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$ROOT/.evaluation_runtime/hf_cache BONEVQA_DATA=$ROOT/.evaluation_runtime/data PYTHONPATH=$R:$ROOT/.evaluation_runtime/deps
export PY=/Users/uyen/Downloads/BoneMedVQA/.venv/bin/python O=reports/thesis_v4_checks_v1 L=reports/thesis_v4_checks_v1/logs
job(){ case $1 in
  cond) [ -f $O/fold_$2/$3/$4.json ] || $PY -m experiments.thesis_v4_checks_v1 --fold $2 --arm $3 --conds $4 --threads 2 >> $L/complete_$2_$3.log 2>&1 || echo "FAIL $*" >> $L/complete_failures.log ;;
  qa) [ -f $O/qa_sample_$3_$4_fold$2.json ] || $PY -m experiments.thesis_v4_qa_sample_v1 $3 $4 $2 >> $L/complete_qa_$2_$3_$4.log 2>&1 || echo "FAIL $*" >> $L/complete_failures.log ;;
esac; echo "$(date +%H:%M:%S) done $*"; }
export -f job
{ for c in reference blank down512 jpeg30 hflip rot90; do for f in 2 3 4; do echo cond $f B $c; done; done
  for c in reference qcond_off iou025; do for f in 3 4; do echo cond $f V4 $c; done; done
  for c in blank down512 down256 jpeg30 blur2 noise10 hflip rot90 crop80 invert; do for f in 2 3 4; do echo cond $f V4 $c; done; done
  for f in 1 2 3 4; do for a in B V4; do for c in blank shuffle; do echo qa $f $a $c; done; done; done
} | caffeinate -i xargs -P 6 -L 1 bash -c 'job "$@"' _
