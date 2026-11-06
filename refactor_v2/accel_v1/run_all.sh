#!/bin/bash
# One command: bash refactor_v2/accel_v1/run_all.sh   (keeps the Mac awake while running)
set -u
ROOT=/Users/uyen/Downloads/BONEVQA; HERE=$ROOT/refactor_v2/accel_v1; mkdir -p "$HERE/logs"
[ -z "${CAFFEINATED:-}" ] && exec env CAFFEINATED=1 caffeinate -dimsu bash "$0" "$@"
exec > >(tee -a "$HERE/logs/run_all.log") 2>&1
echo "== start $(date)"; echo running > "$HERE/STATUS"
bash "$HERE/00_hardware_snapshot.sh"
bash "$HERE/01_setup_accel_venv.sh" || { echo setup_failed > "$HERE/STATUS"; exit 1; }
PY=$ROOT/.evaluation_runtime/accel_venv_v1/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 HF_HOME=$ROOT/.evaluation_runtime/hf_cache BONEVQA_DATA=$ROOT/.evaluation_runtime/data PYTHONPATH=$ROOT/refactor_v2
cd $ROOT/refactor_v2
echo "== MPS parity qualification (torch 2.6)"
$PY experiments/qualify_MPS_zero_update_torch26_v1.py prepare && $PY experiments/qualify_MPS_zero_update_torch26_v1.py run
$PY experiments/qualify_hybrid_vision_MPS_torch26_v1.py prepare && $PY experiments/qualify_hybrid_vision_MPS_torch26_v1.py run
echo "== solo throughput"
for b in 4 8; do $PY -m experiments.benchmark_throughput_accel_v1 --device cpu --batch $b --threads 8 --tag solo; done
for b in 4 8 16; do $PY -m experiments.benchmark_throughput_accel_v1 --device mps --batch $b --threads 4 --tag solo; done
echo "== concurrent CPU job + MPS job"
$PY -m experiments.benchmark_throughput_accel_v1 --device cpu --batch 4 --threads 8 --tag concurrent --repeats 10 &
$PY -m experiments.benchmark_throughput_accel_v1 --device mps --batch 8 --threads 4 --tag concurrent --repeats 10 &
wait
$PY -m experiments.benchmark_throughput_accel_v1 --device cpu --batch 4 --threads 6 --tag twocpu --repeats 8 &
$PY -m experiments.benchmark_throughput_accel_v1 --device cpu --batch 4 --threads 6 --tag twocpu2 --repeats 8 &
wait
bash "$HERE/00_hardware_snapshot.sh"; cp "$HERE/hardware_snapshot.txt" "$HERE/hardware_snapshot_after.txt"
echo done > "$HERE/STATUS"; echo "== done $(date)"
