#!/bin/bash
# Fold 2-4: các điều kiện cốt lõi. Các điều kiện còn lại đã chạy trên fold 0 và 1. MPS không tái lập được dự đoán đã lưu nên chỉ dùng CPU.
ROOT=/Users/uyen/Downloads/BONEVQA; R=$ROOT/refactor_v2; cd "$R" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$ROOT/.evaluation_runtime/hf_cache BONEVQA_DATA=$ROOT/.evaluation_runtime/data PYTHONPATH=$R:$ROOT/.evaluation_runtime/deps
PY=/Users/uyen/Downloads/BoneMedVQA/.venv/bin/python; L=reports/thesis_v4_checks_v1/logs
{ for c in delta_off placebo_boxes annotation_leak iou100 iou000 noise_crops iou050 shuffle; do for f in 2 3 4; do echo "$f V4 $c"; done; done; for f in 2 3 4; do echo "$f B shuffle"; done; } | xargs -P 6 -L 1 bash -c '$0 -m experiments.thesis_v4_checks_v1 --fold $1 --arm $2 --conds $3 --threads 2 >> '$L'/rest_$1_$2.log 2>&1' $PY
