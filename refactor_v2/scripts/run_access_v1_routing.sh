#!/bin/bash
# Phần F của reports/access_v1/PROTOCOL.md. Chờ Phần A và Phần E xong rồi mới chạy (yêu cầu của người dùng: không chèn vào giữa). Chỉ CPU, chạy lại được.
ROOT=/Users/uyen/Downloads/BONEVQA; R=$ROOT/refactor_v2; cd "$R" || exit 1
while pgrep -f "run_access_v1_methods.sh|study_v5.py (worker|job) access_v1" > /dev/null; do sleep 300; done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$ROOT/.evaluation_runtime/hf_cache BONEVQA_DATA=$ROOT/.evaluation_runtime/data PYTHONPATH=$R:$ROOT/.evaluation_runtime/deps
export PY=/Users/uyen/Downloads/BoneMedVQA/.venv/bin/python L=reports/access_v1/logs M=experiments.access_routing_v1; mkdir -p $L
$PY -m $M check > $L/routing_check.log 2>&1 || { echo "FAIL routing check" >> $L/failures.log; exit 1; }
$PY -m $M verify > $L/routing_verify.log 2>&1 || { echo "FAIL routing verify" >> $L/failures.log; exit 1; }
$PY -m $M threshold > $L/routing_threshold.log 2>&1 || echo "FAIL routing threshold" >> $L/failures.log
{ for f in 0 1 2 3 4; do for a in B V4; do for i in 0 1 2 3 4 5; do echo "$f $a old $i router"; echo "$f $a new $i router"; echo "$f $a new $i current"; done; done; done; } | caffeinate -i xargs -P 2 -L 1 bash -c '$PY -m $M run $0 $1 $2 $3 $4 >> $L/routing_run.log 2>&1 || echo "FAIL routing $0 $*" >> $L/failures.log; echo "$(date +%H:%M:%S) routing $0 $*" >> $L/done.log'
$PY -m $M analyze > $L/routing_analyze.log 2>&1 || echo "FAIL routing analyze" >> $L/failures.log
