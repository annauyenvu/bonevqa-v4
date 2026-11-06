#!/bin/bash
# Phần D của reports/access_v1/PROTOCOL.md. Chỉ dùng CPU, chạy lại được.
ROOT=/Users/uyen/Downloads/BONEVQA; R=$ROOT/refactor_v2; cd "$R" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$ROOT/.evaluation_runtime/hf_cache BONEVQA_DATA=$ROOT/.evaluation_runtime/data PYTHONPATH=$R:$ROOT/.evaluation_runtime/deps
export PY=/Users/uyen/Downloads/BoneMedVQA/.venv/bin/python L=reports/access_v1/logs; mkdir -p $L
[ -f reports/access_v1/external/sample.json ] || $PY -m experiments.access_external_v1 sample || exit 1
{ for f in 0 1 2 3 4; do echo "$f B"; echo "$f V4"; done; } | caffeinate -i xargs -P 2 -L 1 bash -c '$PY -m experiments.access_external_v1 run $0 $1 >> $L/ext_$0_$1.log 2>&1 || echo "FAIL ext $0 $1" >> $L/failures.log; echo "$(date +%H:%M:%S) ext $0 $1" >> $L/done.log'
$PY -m experiments.access_external_v1 analyze > $L/ext_analyze.log 2>&1 || echo "FAIL ext analyze" >> $L/failures.log
