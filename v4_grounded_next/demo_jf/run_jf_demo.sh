#!/bin/bash
# Demo API of the final system JF at http://127.0.0.1:8021 (fold: BONEVQA_DEMO_FOLD, default 0)
source /Users/uyen/Downloads/BONEVQA/v4_grounded_next/env.sh
cd /Users/uyen/Downloads/BONEVQA/v4_grounded_next || exit 1
mkdir -p tmp
exec $PY -m uvicorn demo_jf.serve_jf_demo:app --host 127.0.0.1 --port ${BONEVQA_DEMO_PORT:-8021}
