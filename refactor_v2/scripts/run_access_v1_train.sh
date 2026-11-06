#!/bin/bash
# Huấn luyện lại các nhánh đối chứng cho bài IEEE Access (reports/access_v1/PROTOCOL.md). Chạy lại được; job đã xong được bỏ qua.
ROOT=/Users/uyen/Downloads/BONEVQA; cd "$ROOT" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 HF_HOME=$ROOT/.evaluation_runtime/hf_cache BONEVQA_DATA=$ROOT/.evaluation_runtime/data PYTHONPATH=$ROOT/refactor_v2:$ROOT/.evaluation_runtime/deps
PY=$ROOT/.evaluation_runtime/accel_venv_v1/bin/python; S=refactor_v2/experiments/v5/study_v5.py; O=$ROOT/refactor_v2/reports/v5_studies/access_v1
[ -f $O/plan.json ] || $PY $S plan access_v1
mkdir -p $O/fold_0; [ -e $O/fold_0/B_seed43 ] || ln -s $ROOT/refactor_v2/reports/v5_studies/factorial_fold0/fold_0/B_seed43 $O/fold_0/B_seed43
for i in ${1:-1 2}; do caffeinate -i $PY $S worker access_v1 --device mps > $O/worker_$i.log 2>&1 & done
wait
