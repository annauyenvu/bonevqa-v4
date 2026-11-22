"""API parity: p_yes and V4 proposals of the running demo vs the stored JF evaluation (fold 0), on dataset images.
    python3 -I v4_grounded_next/demo_jf/parity_check.py [n]   (server on :8021)"""
import json, subprocess, sys
from pathlib import Path
R = Path("/Users/uyen/Downloads/BONEVQA"); VG = R / "v4_grounded_next/runs"; P = R / "yolo_pilot/datasets"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
JF = "JF_v4cf_yolojoint_btxrd_r0.02"
fa = {r["image_id"]: r["p_final"] for r in json.load(open(VG / f"yolo_ablation/fold_0/{JF}_test.json"))["rows"]}
ref = json.load(open(VG / f"joint/fold_0/ref_{JF}_test.json"))["endpoints"]
tu = {r["image_id"]: r["plain"] for r in ref["tumor"]}; ma = {r["image_id"]: r["plain"] for r in ref["malignancy"]}
fprops = json.load(open(VG / "crossfit_yolo_joint_sweep/r0.02/fold_0/proposals_fold0.json"))["test"]
bprops = json.load(open(VG / "crossfit_yolo_joint_sweep/r0.10/btxrd_fold0/proposals_btxrd_insample.json"))["test"]
def ask(path, q):
    out = subprocess.run(["curl", "-s", "-F", f"image=@{path}", "-F", f"question={q}", "localhost:8021/api/ask"], capture_output=True, text=True).stdout
    return json.loads(out)
rows = []
for iid in sorted(fa)[:: max(1, len(fa) // n)][:n]:
    d = ask(P / f"fa_fold0/images/test/{iid}.jpg", "Is there a fracture in this X-ray?")
    rows.append(("fracture", iid, d["p_yes"], fa[iid], len(d["proposals"]), len(fprops.get(iid, []))))
for iid in sorted(ma)[:: max(1, len(ma) // n)][:n]:
    f = next((P / "btxrd_pilot/images/test").glob(f"{iid}.*"))
    d = ask(f, "Is there a bone tumor in this X-ray?"); rows.append(("tumor", iid, d["p_yes"], tu[iid], len(d["proposals"]), len(bprops.get(iid, []))))
    d = ask(f, "Is the bone lesion malignant?"); rows.append(("malignancy", iid, d["p_yes"], ma[iid], len(d["proposals"]), len(bprops.get(iid, []))))
for r in rows: print(r)
dp = [abs(a - b) for _, _, a, b, _, _ in rows]
print(json.dumps({"rows": len(rows), "max_abs_dp": max(dp), "mean_abs_dp": sum(dp) / len(dp),
                  "proposal_count_match": sum(x == y for *_, x, y in rows) / len(rows)}))
