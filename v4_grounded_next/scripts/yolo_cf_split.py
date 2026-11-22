"""Inner split for cross-fitting a YOLOv8 detector on the TRAIN role of an outer fold.

Same rule as ``cross_fit_proposals.plan`` (StratifiedGroupKFold on fractured x
component, seed 20261111) but K = 3, because one YOLOv8s detector costs about
2.7 h on this machine instead of 20 min for the V4 Faster R-CNN recipe.

    source v4_grounded_next/env.sh
    $PY -m scripts.yolo_cf_split --fold 0
"""
from __future__ import annotations

import argparse

from vg.paths import RUNS, component_lookup, dump_json, image_table, manifest_fold, read_json, sha256

K_INNER = 3
SPLIT_SEED = 20261111


def main(fold):
    from sklearn.model_selection import StratifiedGroupKFold
    roles = manifest_fold(fold)
    table = image_table(read_json(roles["train"]))
    lookup = component_lookup()
    ids = sorted(table)
    groups = [str(lookup.get(i, f"solo:{i}")) for i in ids]
    y = [int(table[i]["fractured"]) for i in ids]
    gid = {g: n for n, g in enumerate(sorted(set(groups)))}
    sgkf = StratifiedGroupKFold(n_splits=K_INNER, shuffle=True, random_state=SPLIT_SEED)
    inner = {}
    for k, (_, held) in enumerate(sgkf.split(ids, y, [gid[g] for g in groups])):
        inner[str(k)] = sorted(ids[j] for j in held)
    where = {}
    for k, held in inner.items():
        for i in held:
            g = groups[ids.index(i)]
            assert where.setdefault(g, k) == k, f"component {g} split across inner folds"
    assert sorted(i for v in inner.values() for i in v) == ids
    out = {"outer_fold": str(fold), "train_manifest": str(roles["train"]), "train_sha256": sha256(roles["train"]),
           "k_inner": K_INNER, "seed": SPLIT_SEED, "images": len(ids), "inner": inner,
           "inner_stats": {k: {"images": len(v), "fractured": sum(int(table[i]["fractured"]) for i in v)}
                           for k, v in inner.items()}}
    dump_json(RUNS / "crossfit_yolo" / f"fold_{fold}" / "inner_split.json", out, exclusive=True)
    print(out["inner_stats"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="0")
    main(ap.parse_args().fold)
