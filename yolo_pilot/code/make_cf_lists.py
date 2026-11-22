"""Image lists for cross-fitting the FracAtlas-only YOLOv8 detector on TRAIN of a fold.

    python3 -I code/make_cf_lists.py 0

Reads the inner split written by v4_grounded_next/scripts/yolo_cf_split.py.  For inner
fold k: ``inner{k}_train_os3.txt`` holds every TRAIN image outside inner k once and the
fractured ones three times (the recipe of the outer detector), ``inner{k}_heldout.txt``
holds the images of inner k, which that detector never sees.
"""
import json
import sys
from pathlib import Path

ROOT = Path("/Users/uyen/Downloads/BONEVQA")


def main(fold):
    ds = ROOT / f"yolo_pilot/datasets/fa_fold{fold}"
    split = json.loads((ROOT / f"v4_grounded_next/runs/crossfit_yolo/fold_{fold}/inner_split.json").read_text())
    gt = json.loads((ds / "gt.json").read_text())
    files = {p.stem: p for p in (ds / "images/train").iterdir()}
    train_ids = sorted(i for i, g in gt.items() if g["role"] == "train")
    assert sorted(i for v in split["inner"].values() for i in v) == train_ids == sorted(files)
    out = ds / "cf"
    out.mkdir(exist_ok=False)
    stats = {}
    for k, held in split["inner"].items():
        held_set = set(held)
        lines = []
        for i in train_ids:
            if i in held_set:
                continue
            lines += [str(files[i])] * (3 if gt[i]["fractured"] else 1)
        (out / f"inner{k}_train_os3.txt").write_text("\n".join(lines) + "\n")
        (out / f"inner{k}_heldout.txt").write_text("\n".join(str(files[i]) for i in held) + "\n")
        (out / f"data_cf{k}.yaml").write_text(
            f"path: {ds}\ntrain: cf/inner{k}_train_os3.txt\nval: images/val\nnames:\n  0: fracture\n")
        stats[k] = {"train_images": len(train_ids) - len(held), "train_entries": len(lines), "heldout": len(held),
                    "heldout_fractured": sum(gt[i]["fractured"] for i in held)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
