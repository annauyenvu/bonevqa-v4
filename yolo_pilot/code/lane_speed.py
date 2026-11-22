"""Speed of the two lanes against the single-job baselines measured on fold 0 (read-only).

    python3 code/lane_speed.py            # one line per lane
    python3 code/lane_speed.py --watch    # every 10 min; exits (printing why) when a lane is >15% slower than its
                                          # baseline on two checks in a row, or a lane has written nothing for 40 min
V4: seconds per update over the last ~30 min, from progress.json of the running training.
    Baselines, alone on the machine: train_joint 5.25 s, train_arm 5.1 s (fold 0 runs; seed 44 before the lanes: 5.24).
YOLOv8: mean seconds of the last 3 epochs, against the same detector of fold 0 (e.g. cfj1_f2 -> cfj1).
"""
import csv
import glob
import json
import os
import subprocess
import sys
import time

ARMS = "/Users/uyen/Downloads/BONEVQA/v4_grounded_next/runs/arms"
YOLO = "/Users/uyen/Downloads/BONEVQA/yolo_pilot/runs/yolo"
STATE = "/Users/uyen/Downloads/BONEVQA/yolo_pilot/runs/lanes/speed_samples.json"
BASE_V4 = {"joint": 5.25, "arm": 5.1}
SLOW = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--slow=")), 1.15))
WINDOW = 1800


def epoch_times(path):
    t = [float(r["time"]) for r in csv.DictReader(open(path))]
    return [b - a for a, b in zip([0.0] + t, t)]


def v4():
    live = [p for p in glob.glob(f"{ARMS}/**/progress.json", recursive=True)
            if not os.path.exists(os.path.join(os.path.dirname(p), "completed.json")) and time.time() - os.path.getmtime(p) < 2400]
    if not live:
        return None
    p = max(live, key=os.path.getmtime)
    d = json.load(open(p))
    kind = "joint" if "joint_fold" in p else "arm"
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    s = [x for x in st.get(p, []) if time.time() - x[0] < 4 * WINDOW] + [[time.time(), d["updates"], d["seconds"]]]
    st[p] = s
    st = {k: v for k, v in st.items() if k == p or time.time() - v[-1][0] < 86400}
    json.dump(st, open(STATE, "w"))
    old = [x for x in s if time.time() - x[0] >= WINDOW * 0.5] or s[:1]
    a, b = old[-1], s[-1]
    rate = (b[2] - a[2]) / (b[1] - a[1]) if b[1] > a[1] else d["seconds"] / max(d["updates"], 1)
    run = p.replace(ARMS + "/", "").replace("/progress.json", "")
    left_h = (d["stop"] - d["updates"]) * rate / 3600
    return {"lane": "V4 (CPU)", "run": run, "progress": f"{d['updates']}/{d['stop']}", "rate": rate, "base": BASE_V4[kind],
            "unit": "s/bước", "left_h": left_h, "age_min": (time.time() - os.path.getmtime(p)) / 60}


def yolo():
    # a stopped detector folder has no "done" marker either: only report while a YOLOv8 training runs
    if subprocess.run(["pgrep", "-f", r"code/train_yolo\.py"], capture_output=True).returncode != 0:
        return None
    live =[p for p in glob.glob(f"{YOLO}/*_f[1-9]/results.csv") if not os.path.exists(os.path.join(os.path.dirname(p), "done"))]
    if not live:
        return None
    p = max(live, key=os.path.getmtime)
    name = os.path.basename(os.path.dirname(p))
    et = epoch_times(p)
    if not et:
        return {"lane": "YOLOv8 (GPU)", "run": name, "progress": "0/60", "rate": None, "base": None, "unit": "s/epoch", "left_h": None,
                "age_min": (time.time() - os.path.getmtime(p)) / 60}
    ref = f"{YOLO}/{name.rsplit('_f', 1)[0]}/results.csv"
    base_et = epoch_times(ref) if os.path.exists(ref) else []
    n = len(et)
    rate = sum(et[-3:]) / len(et[-3:])
    base = sorted(base_et)[len(base_et) // 2] if base_et else None   # median: fold-0 epochs include machine sleeps
    return {"lane": "YOLOv8 (GPU)", "run": name, "progress": f"{n}/60", "rate": rate, "base": base, "unit": "s/epoch",
            "left_h": (60 - n) * rate / 3600, "age_min": (time.time() - os.path.getmtime(p)) / 60}


def mem():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.strip().splitlines()
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    return (out[-1].split(":")[-1].strip() if out else "?") + " free, swap " + sw.split("used =")[-1].split("free")[0].strip()


def line(r):
    if r is None:
        return None
    ratio = r["rate"] / r["base"] if r["rate"] and r["base"] else None
    s = f"{r['lane']}: {r['run']} {r['progress']}"
    if r["rate"]:
        s += f", {r['rate']:.1f} {r['unit']}"
    if r["base"]:
        s += f" (một mình: {r['base']:.1f}; {ratio:.2f}x)"
    if r["left_h"] is not None:
        s += f", còn ~{r['left_h']:.1f} giờ"
    return s, ratio


def main():
    watch = "--watch" in sys.argv
    strikes = {}
    while True:
        rows = [x for x in (v4(), yolo()) if x]
        msgs = [line(r) for r in rows]
        print(time.strftime("%F %T"), "|", " | ".join(m[0] for m in msgs), "| RAM", mem(), flush=True)
        if not watch:
            return
        alarm = []
        for r, (txt, ratio) in zip(rows, msgs):
            strikes[r["lane"]] = strikes.get(r["lane"], 0) + 1 if ratio and ratio > SLOW else 0
            if strikes[r["lane"]] >= 2:
                alarm.append(f"CHẬM: {txt}")
            if r["age_min"] > 40 and not (r["lane"].startswith("V4") and os.path.exists(os.path.dirname(STATE) + "/v4_paused")):
                alarm.append(f"KHÔNG CẬP NHẬT {r['age_min']:.0f} phút: {r['run']}")
        if alarm:
            print("\n".join(alarm), flush=True)
            return
        time.sleep(600)


if __name__ == "__main__":
    main()
