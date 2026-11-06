#!/bin/bash
# Phần B của reports/access_v1/PROTOCOL.md: diễn đạt lại câu hỏi và baseline chỉ dùng ảnh. Chỉ dùng CPU, chạy lại được.
ROOT=/Users/uyen/Downloads/BONEVQA; R=$ROOT/refactor_v2; cd "$R" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$ROOT/.evaluation_runtime/hf_cache BONEVQA_DATA=$ROOT/.evaluation_runtime/data PYTHONPATH=$R:$ROOT/.evaluation_runtime/deps
export PY=/Users/uyen/Downloads/BoneMedVQA/.venv/bin/python L=reports/access_v1/logs; mkdir -p $L
{ $PY -m experiments.access_image_baselines_v1 clip > $L/clip.log 2>&1 || echo FAIL clip >> $L/failures.log; } &
{ for f in 0 1 2 3 4; do for a in B V4; do for i in 0 1 2 3 4 5; do echo "para $f $a $i"; done; done; done; for f in 0 1 2 3 4; do echo "mob $f"; done; } | caffeinate -i xargs -P 2 -L 1 bash -c 'if [ $0 = para ]; then $PY -m experiments.access_paraphrase_v1 $1 $2 $3 >> $L/para.log 2>&1 || echo "FAIL $0 $*" >> $L/failures.log; else $PY -m experiments.access_image_baselines_v1 mobilenet $1 >> $L/mob_$1.log 2>&1 || echo "FAIL $0 $*" >> $L/failures.log; fi; echo "$(date +%H:%M:%S) $0 $*" >> $L/done.log'
wait
