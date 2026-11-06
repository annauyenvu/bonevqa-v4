#!/bin/bash
# New venv = old BoneMedVQA venv packages, but torch 2.6.0 for newer MPS kernels. Old venv untouched.
set -euo pipefail
OLD=/Users/uyen/Downloads/BoneMedVQA/.venv/bin/python
NEW=/Users/uyen/Downloads/BONEVQA/.evaluation_runtime/accel_venv_v1
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -x "$NEW/bin/python" ] && { echo "venv exists: $NEW"; exit 0; }
BASE=$("$OLD" -c "import sys;print(sys._base_executable)")
"$BASE" -m venv "$NEW"
"$OLD" -m pip freeze --exclude-editable | grep -v -E '^(torch|torchvision|torchaudio)(==| @)' > "$HERE/old_venv_freeze_without_torch.txt"
"$NEW/bin/pip" install -q -U pip
"$NEW/bin/pip" install torch==2.6.0 torchvision==0.21.0
"$NEW/bin/pip" install -r "$HERE/old_venv_freeze_without_torch.txt" || "$NEW/bin/pip" install --no-deps -r "$HERE/old_venv_freeze_without_torch.txt"
"$NEW/bin/pip" freeze > "$HERE/accel_venv_freeze.txt"
"$NEW/bin/python" -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__,'mps',torch.backends.mps.is_available())"
