"""Training lists and data files for the joint YOLOv8 detectors of one outer fold (k >= 1).

    python3 -I code/prep_fold_yaml.py 1

Run after prep_fracatlas.py k and make_cf_lists.py k.  Same recipe as fold 0:
  datasets/fa_fold<k>/train_os3.txt              every TRAIN image once, fractured ones three times
  datasets/fa_fold<k>/data_fa_btxrd_os3.yaml     outer joint detector: FracAtlas TRAIN (os3) + BTXRD TRAIN
  datasets/fa_fold<k>/cf/data_cfj<i>.yaml        inner joint detector i: inner-i TRAIN list (os3) + BTXRD TRAIN
Validation (for ultralytics' best.pt) = FracAtlas model_val of the fold + BTXRD val, as for fold 0.
"""
import json
import sys
from pathlib import Path

DS = Path("/Users/uyen/Downloads/BONEVQA/yolo_pilot/datasets")


def main(fold):
    assert fold != 0, "fold 0 files already exist under their original names"
    d = DS / f"fa_fold{fold}"
    gt = json.loads((d / "gt.json").read_text())
    files = {p.stem: p for p in (d / "images/train").iterdir()}
    train = sorted(i for i, g in gt.items() if g["role"] == "train")
    assert sorted(files) == train
    lines = [str(files[i]) for i in train for _ in range(3 if gt[i]["fractured"] else 1)]
    (d / "train_os3.txt").write_text("\n".join(lines) + "\n")
    (d / "data_fa_os3.yaml").write_text(f"path: {d}\ntrain: train_os3.txt\nval: images/val\ntest: images/test\nnames:\n  0: fracture\n")
    joint = lambda train_list: (f"path: {DS}\ntrain:\n  - fa_fold{fold}/{train_list}\n  - btxrd_pilot/images/train\n"
                                f"val:\n  - fa_fold{fold}/images/val\n  - btxrd_pilot/images/val\nnames:\n  0: fracture\n  1: bone_lesion\n")
    (d / "data_fa_btxrd_os3.yaml").write_text(joint("train_os3.txt"))
    for i in range(3):
        assert (d / f"cf/inner{i}_train_os3.txt").exists()
        (d / f"cf/data_cfj{i}.yaml").write_text(joint(f"cf/inner{i}_train_os3.txt"))
    print(json.dumps({"fold": fold, "train_images": len(train), "train_entries": len(lines)}))


if __name__ == "__main__":
    main(int(sys.argv[1]))
