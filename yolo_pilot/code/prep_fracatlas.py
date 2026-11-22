"""FracAtlas fold -> YOLO detection dataset (class 0 = fracture).

Reads the sealed fold manifests read-only.  Images are symlinked when the file is
already canonical (EXIF orientation 1) and re-saved as PNG (EXIF applied) otherwise,
so that what YOLO reads is the same canonical raster the V4 pipeline uses.  GT boxes
in the manifests are raw-file coordinates; they are mapped to the canonical frame
with the same table as v4_grounded_next/vg/geometry.exif_point_fn.
"""
import ast
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
CV = ROOT / ".evaluation_runtime/data/fracatlas_component_cv_v3"
FRACATLAS = ROOT / ".evaluation_runtime/data/fracatlas/FracAtlas"
OUT = ROOT / "yolo_pilot/datasets"
ROLES = {"train": "train", "model_val": "val", "calibration": "cal", "test": "test"}


def exif_point_fn(orientation, w, h):
    table = {
        1: lambda x, y: (x, y), 2: lambda x, y: (w - x, y), 3: lambda x, y: (w - x, h - y),
        4: lambda x, y: (x, h - y), 5: lambda x, y: (y, x), 6: lambda x, y: (h - y, x),
        7: lambda x, y: (h - y, w - x), 8: lambda x, y: (y, w - x),
    }
    return table.get(int(orientation), table[1])


def canonical_boxes(boxes, orientation, raw_size):
    fn = exif_point_fn(orientation, *raw_size)
    out = []
    for b in boxes or []:
        x1, y1, x2, y2 = map(float, b[:4])
        xs, ys = zip(*[fn(x1, y1), fn(x1, y2), fn(x2, y1), fn(x2, y2)])
        out.append([min(xs), min(ys), max(xs), max(ys)])
    return out


def main(fold):
    dest = OUT / f"fa_fold{fold}"
    assert not dest.exists(), f"{dest} exists"
    gt, stats = {}, {}
    for role, short in ROLES.items():
        records = json.loads((CV / f"fold_{fold}/vqa_{role}.json").read_text())
        table = {}
        for r in records:
            table.setdefault(r["image_id"], r)
        (dest / "images" / short).mkdir(parents=True)
        (dest / "labels" / short).mkdir(parents=True)
        n_box = n_rot = n_fx = 0
        for iid in sorted(table):
            r = table[iid]
            src = FRACATLAS / r["image"]
            with Image.open(src) as im:
                orientation = int(im.getexif().get(274, 1) or 1)
                raw_size = im.size
                if orientation != 1:
                    canon = ImageOps.exif_transpose(im).convert("RGB")
                    canon.save(dest / "images" / short / f"{iid}.png")
                    size = canon.size
                    n_rot += 1
                else:
                    (dest / "images" / short / f"{iid}{src.suffix}").symlink_to(src)
                    size = raw_size
            raw = r["boxes"]
            if isinstance(raw, str):
                raw = ast.literal_eval(raw) if raw not in ("None", "") else []
            raw = raw or []
            boxes = canonical_boxes(raw, orientation, raw_size)
            w, h = size
            lines = []
            for x1, y1, x2, y2 in boxes:
                x1, x2 = max(0.0, min(w, x1)), max(0.0, min(w, x2))
                y1, y2 = max(0.0, min(h, y1)), max(0.0, min(h, y2))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (dest / "labels" / short / f"{iid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            fractured = str(r["fractured"]) == "1"
            assert fractured == bool(boxes), (iid, r["fractured"], boxes)
            gt[iid] = {"role": short, "size": [w, h], "fractured": int(fractured), "boxes": boxes,
                       "hardware": int(str(r.get("hardware", "0")) == "1"), "orientation": orientation}
            n_box += len(boxes)
            n_fx += fractured
        stats[short] = {"images": len(table), "fractured": n_fx, "boxes": n_box, "exif_rotated": n_rot}
    (dest / "gt.json").write_text(json.dumps(gt))
    (dest / "stats.json").write_text(json.dumps(stats, indent=2))
    (dest / "data_fa.yaml").write_text(
        f"path: {dest}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: fracture\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
