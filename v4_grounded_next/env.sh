# Source before every command:  source v4_grounded_next/env.sh
# Keeps every byproduct (bytecode, HF temp files, TMP) out of refactor_v2/.
export ROOT=/Users/uyen/Downloads/BONEVQA
export VG=$ROOT/v4_grounded_next
export PY=/Users/uyen/Downloads/BoneMedVQA/.venv/bin/python   # torch 2.2.2, the sealed V4 interpreter
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$ROOT/.evaluation_runtime/hf_cache
export BONEVQA_DATA=$ROOT/.evaluation_runtime/data
# refactor_v2 is imported READ-ONLY. Bytecode goes to our own prefix, never to
# refactor_v2/**/__pycache__.
export PYTHONPYCACHEPREFIX=$VG/.pycache
export PYTHONDONTWRITEBYTECODE=1
# refactor_v2/src/utils.setup_env_dirs() does os.makedirs(setdefault(...)) at
# import time; pre-setting these means it creates nothing under refactor_v2.
export TMP=$VG/tmp TEMP=$VG/tmp TMPDIR=$VG/tmp PIP_CACHE_DIR=$VG/tmp/pip
export PYTHONPATH=$VG:$ROOT/refactor_v2:$ROOT/.evaluation_runtime/deps
export PYTORCH_ENABLE_MPS_FALLBACK=0   # fail loudly instead of silent CPU fallback
export OMP_NUM_THREADS=8
cd $ROOT   # never run with cwd inside refactor_v2 (relative 'D:/...' dirs)
