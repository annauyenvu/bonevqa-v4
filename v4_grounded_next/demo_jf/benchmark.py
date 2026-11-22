"""Latency of the JF demo API over HTTP (server on :8021): every example image x 3 repeats per request type.
    python3 -I v4_grounded_next/demo_jf/benchmark.py  -> demo_jf/benchmark.json"""
import json, statistics, subprocess, time, platform
from pathlib import Path
H = Path(__file__).resolve().parent; EX = sorted((H / "web/examples").glob("*.jpg")); REP = 3
def call(args):
    t = time.perf_counter(); out = subprocess.run(["curl", "-s", *args], capture_output=True, text=True).stdout
    return (time.perf_counter() - t) * 1000, json.loads(out)
kinds = {"detect": None, "fracture": "Is there a fracture in this X-ray?", "tumor": "Is there a bone tumor in this X-ray?",
         "open": "Which body region is shown in this X-ray?", "diagnosis": "What is the most likely diagnosis?"}
res = {}
for k, q in kinds.items():
    ms = []
    for f in EX:
        for _ in range(REP):
            args = ["-F", f"image=@{f}", "localhost:8021/api/detect"] if q is None else ["-F", f"image=@{f}", "-F", f"question={q}", "localhost:8021/api/ask"]
            m, d = call(args); assert "detail" not in d, d; ms.append(m)
    ms.sort(); res[k] = {"n": len(ms), "p50": round(statistics.median(ms)), "p95": round(ms[int(0.95 * (len(ms) - 1))]), "max": round(ms[-1])}
health = call(["localhost:8021/api/health"])[1]
out = {"time": time.strftime("%Y-%m-%d %H:%M"), "machine": platform.platform(), "health": health, "images": len(EX), "repeats": REP, "latency_ms": res}
(H / "benchmark.json").write_text(json.dumps(out, ensure_ascii=False, indent=1)); print(json.dumps(res))
