#!/bin/bash
# Local V4 thesis demo at http://127.0.0.1:8020
ROOT=/Users/uyen/Downloads/BONEVQA; R=$ROOT/refactor_v2; cd "$R" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$ROOT/.evaluation_runtime/hf_cache BONEVQA_DATA=$ROOT/.evaluation_runtime/data PYTHONPATH=$R:$ROOT/.evaluation_runtime/deps
exec /Users/uyen/Downloads/BoneMedVQA/.venv/bin/python -m uvicorn experiments.serve_v4_thesis_demo:app --host 127.0.0.1 --port 8020
