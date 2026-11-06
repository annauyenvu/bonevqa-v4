@echo off
cd /d "%~dp0.."
set HF_HOME=D:\BaoYen_work\hf_cache
set TMP=D:\BaoYen_work\tmp
set TEMP=D:\BaoYen_work\tmp
set PIP_CACHE_DIR=D:\BaoYen_work\pip_cache
if "%BONEVQA_CKPT%"=="" set BONEVQA_CKPT=D:\BaoYen_work\checkpoints\best.pt
echo Khoi dong BoneVQA-Prompt tai http://localhost:8000
"D:\BaoYen_work\venv\Scripts\python.exe" -m uvicorn app.api:app --host 0.0.0.0 --port 8000
