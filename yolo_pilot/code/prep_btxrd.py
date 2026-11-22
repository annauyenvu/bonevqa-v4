"""BTXRD -> YOLO detection labels (class 1 = bone_lesion), pilot split.

* Centre 1 (three hospitals, 2,938 images) is split train / val / test by
  pseudo-patient.  BTXRD ships no patient id, so a pseudo-patient is a run of
  consecutive image ids that share centre, age, gender, tumour flag, subtype flags
  and limb region (frontal / lateral / oblique views of one lesion are consecutive).
  This is a heuristic, not a verified patient key.
* Centres 2 and 3 (Radiopaedia, MedPix; 808 images) are never used for training:
  they go to an ``external`` role.
* Lesion boxes are the Labelme rectangles; a polygon whose bounding box overlaps no
  rectangle (IoU < 0.3) is added as its bounding box.
* Fractures are NOT annotated in BTXRD.  Every BTXRD image therefore has no class-0
  label, whether or not it shows a (pathological) fracture.
"""
import json
import random
from collections import Counter
from pathlib import Path

import openpyxl
from PIL import Image

ROOT = Path("/Users/uyen/Downloads/BONEVQA/yolo_pilot")
SRC = ROOT / "data_raw/btxrd/BTXRD"
DEST = ROOT / "datasets/btxrd_pilot"
SUBTYPES = ["osteochondroma", "multiple osteochondromas", "simple bone cyst", "giant cell tumor", "osteofibroma",
            "synovial osteochondroma", "other bt", "osteosarcoma", "other mt"]
SEED = 20261114
FRACTIONS = {"train": 0.70, "val": 0.10, "test": 0.20}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def lesion_boxes(stem):
    p = SRC / "Annotations" / f"{stem}.json"
    if not p.exists():
        return []
    shapes = json.loads(p.read_text())["shapes"]
    rects, polys = [], []
    for s in shapes:
        xs = [float(pt[0]) for pt in s["points"]]
        ys = [float(pt[1]) for pt in s["points"]]
        box = [min(xs), min(ys), max(xs), max(ys), s["label"]]
        (rects if s["shape_type"] == "rectangle" else polys).append(box)
    out = list(rects)
    for pb in polys:
        if all(iou(pb, r) < 0.3 for r in rects):
            out.append(pb)
    return out


def main():
    assert not DEST.exists(), f"{DEST} exists"
    ws = openpyxl.load_workbook(SRC / "dataset.xlsx", read_only=True, data_only=True)["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))
    hdr = {h: i for i, h in enumerate(rows[0])}
    recs = []
    for r in rows[1:]:
        recs.append({"file": r[hdr["image_id"]], "center": int(r[hdr["center"]]), "age": str(r[hdr["age"]]),
                     "gender": r[hdr["gender"]], "tumor": int(r[hdr["tumor"]]), "malignant": int(r[hdr["malignant"]]),
                     "subtype": next((s for s in SUBTYPES if r[hdr[s]] == 1), None),
                     "region": tuple(int(r[hdr[k]]) for k in ("upper limb", "lower limb", "pelvis"))})
    recs.sort(key=lambda x: x["file"])
    # pseudo-patients
    gid, prev = -1, None
    for x in recs:
        key = (x["center"], x["age"], x["gender"], x["tumor"], x["subtype"], x["region"])
        if key != prev:
            gid += 1
            prev = key
        x["group"] = gid
    groups = {}
    for x in recs:
        groups.setdefault(x["group"], []).append(x)
    rng = random.Random(SEED)
    role_of = {}
    for tumor in (0, 1):
        gids = sorted(g for g, v in groups.items() if v[0]["center"] == 1 and v[0]["tumor"] == tumor)
        rng.shuffle(gids)
        total = sum(len(groups[g]) for g in gids)
        acc, bounds = 0, (FRACTIONS["train"], FRACTIONS["train"] + FRACTIONS["val"])
        for g in gids:
            frac = acc / total
            role_of[g] = "train" if frac < bounds[0] else "val" if frac < bounds[1] else "test"
            acc += len(groups[g])
    for g, v in groups.items():
        if v[0]["center"] != 1:
            role_of[g] = "external"

    gt, stats = {}, {r: Counter() for r in ("train", "val", "test", "external")}
    for role in stats:
        (DEST / "images" / role).mkdir(parents=True)
        (DEST / "labels" / role).mkdir(parents=True)
    for x in recs:
        role = role_of[x["group"]]
        src = SRC / "images" / x["file"]
        stem = Path(x["file"]).stem
        with Image.open(src) as im:
            orientation = int(im.getexif().get(274, 1) or 1)
            w, h = im.size
        assert orientation == 1, (x["file"], orientation)
        boxes = lesion_boxes(stem)
        assert bool(boxes) == bool(x["tumor"]), (x["file"], len(boxes), x["tumor"])
        lines, kept = [], []
        for x1, y1, x2, y2, label in boxes:
            x1, x2 = max(0.0, min(w, x1)), max(0.0, min(w, x2))
            y1, y2 = max(0.0, min(h, y1)), max(0.0, min(h, y2))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            lines.append(f"1 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            kept.append([x1, y1, x2, y2, label])
        name = f"BT_{stem}"
        (DEST / "images" / role / f"{name}{src.suffix}").symlink_to(src)
        (DEST / "labels" / role / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        gt[name] = {"role": role, "size": [w, h], "tumor": x["tumor"], "malignant": x["malignant"],
                    "subtype": x["subtype"], "center": x["center"], "group": x["group"], "boxes": kept}
        s = stats[role]
        s["images"] += 1
        s["tumor"] += x["tumor"]
        s["malignant"] += x["malignant"]
        s["boxes"] += len(kept)
    (DEST / "gt.json").write_text(json.dumps(gt))
    summary = {"seed": SEED, "pseudo_patients": len(groups),
               "largest_group": max(len(v) for v in groups.values()),
               "roles": {r: dict(c) for r, c in stats.items()}}
    (DEST / "stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
