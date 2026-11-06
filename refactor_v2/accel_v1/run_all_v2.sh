#!/bin/bash
# Fixes v1: pin torch 2.6.0, install open_clip at the exact old-venv version, run from BONEVQA root.
set -u
ROOT=/Users/uyen/Downloads/BONEVQA; HERE=$ROOT/refactor_v2/accel_v1; mkdir -p "$HERE/logs"
[ -z "${CAFFEINATED:-}" ] && exec env CAFFEINATED=1 caffeinate -dimsu bash "$0" "$@"
exec > >(tee -a "$HERE/logs/run_all_v2.log") 2>&1
echo "== start $(date)"; echo running > "$HERE/STATUS_v2"
OLD=/Users/uyen/Downloads/BoneMedVQA/.venv/bin/python; PY=$ROOT/.evaluation_runtime/accel_venv_v1/bin/python
echo "== old venv open_clip / timm / numpy"
"$OLD" -m pip freeze | grep -i -E "open.clip|timm|numpy|sympy|^-e" | tee "$HERE/old_venv_special.txt"
"$OLD" -c "import open_clip,sys;print('open_clip',open_clip.__version__,open_clip.__file__)" | tee -a "$HERE/old_venv_special.txt"
OCV=$("$OLD" -c "import open_clip;print(open_clip.__version__)")
printf 'torch==2.6.0\ntorchvision==0.21.0\nnumpy==1.26.4\n' > "$HERE/constraints.txt"
"$PY" -m pip install -q "torch==2.6.0" "torchvision==0.21.0" "numpy==1.26.4" "sympy==1.13.1" -c "$HERE/constraints.txt"
"$PY" -m pip install -q "open_clip_torch==$OCV" -c "$HERE/constraints.txt" || {
  SRC=$("$OLD" -c "import open_clip,os;print(os.path.dirname(os.path.dirname(open_clip.__file__)))")
  echo "pip open_clip_torch==$OCV failed; installing from $SRC"; "$PY" -m pip install -q --no-deps -e "$SRC" || "$PY" -m pip install -q --no-deps "$SRC"; }
"$PY" -m pip freeze > "$HERE/accel_venv_freeze_v2.txt"
"$PY" -c "import torch,transformers,open_clip,numpy;print('torch',torch.__version__,'transformers',transformers.__version__,'open_clip',open_clip.__version__,'numpy',numpy.__version__,'mps',torch.backends.mps.is_available())" || { echo env_failed > "$HERE/STATUS_v2"; exit 1; }
cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 HF_HOME=$ROOT/.evaluation_runtime/hf_cache BONEVQA_DATA=$ROOT/.evaluation_runtime/data PYTHONPATH=$ROOT/refactor_v2
rm -rf refactor_v2/reports/MPS_zero_update_qualification_torch26_v1 refactor_v2/reports/hybrid_vision_MPS_qualification_torch26_v1
echo "== MPS parity qualification (torch 2.6)"
$PY refactor_v2/experiments/qualify_MPS_zero_update_torch26_v1.py prepare && $PY refactor_v2/experiments/qualify_MPS_zero_update_torch26_v1.py run
$PY refactor_v2/experiments/qualify_hybrid_vision_MPS_torch26_v1.py prepare && $PY refactor_v2/experiments/qualify_hybrid_vision_MPS_torch26_v1.py run
echo "== solo throughput"
rm -rf "$HERE/bench"
for b in 4 8; do $PY -m experiments.benchmark_throughput_accel_v1 --device cpu --batch $b --threads 8 --tag solo; done
for b in 4 8 16; do $PY -m experiments.benchmark_throughput_accel_v1 --device mps --batch $b --threads 4 --tag solo; done
echo "== concurrent CPU job + MPS job"
$PY -m experiments.benchmark_throughput_accel_v1 --device cpu --batch 4 --threads 8 --tag concurrent --repeats 10 &
$PY -m experiments.benchmark_throughput_accel_v1 --device mps --batch 8 --threads 4 --tag concurrent --repeats 10 &
wait
$PY -m experiments.benchmark_throughput_accel_v1 --device cpu --batch 4 --threads 6 --tag twocpu_a --repeats 8 &
$PY -m experiments.benchmark_throughput_accel_v1 --device cpu --batch 4 --threads 6 --tag twocpu_b --repeats 8 &
wait
echo done > "$HERE/STATUS_v2"; echo "== done $(date)"
