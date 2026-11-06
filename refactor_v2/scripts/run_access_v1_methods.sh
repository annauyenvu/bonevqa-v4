#!/bin/bash
# Phần E của reports/access_v1/PROTOCOL.md. Chờ worker của Phần A kết thúc (máy chỉ đủ RAM cho một worker MPS), rồi chạy. Job đã xong được bỏ qua.
ROOT=/Users/uyen/Downloads/BONEVQA; cd "$ROOT" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 HF_HOME=$ROOT/.evaluation_runtime/hf_cache BONEVQA_DATA=$ROOT/.evaluation_runtime/data PYTHONPATH=$ROOT/refactor_v2:$ROOT/.evaluation_runtime/deps
PY=$ROOT/.evaluation_runtime/accel_venv_v1/bin/python; S=refactor_v2/experiments/v5/study_v5.py; O=$ROOT/refactor_v2/reports/v5_studies/access_v1_methods
while pgrep -f "study_v5.py (worker|job) access_v1( |$)" > /dev/null; do sleep 300; done
mkdir -p "$O"; [ -f "$O/plan.json" ] || $PY $S plan access_v1_methods || exit 1
mkdir -p "$O/fold_0"; for a in V P L; do [ -e "$O/fold_0/${a}_seed42" ] || ln -s "$ROOT/refactor_v2/reports/v5_studies/factorial_fold0/fold_0/${a}_seed42" "$O/fold_0/${a}_seed42"; done
caffeinate -i $PY $S worker access_v1_methods --device mps > "$O/worker.log" 2>&1
