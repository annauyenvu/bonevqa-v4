"""Path contract and write guard.

Every write in this workspace goes through :func:`dump_json` / :func:`guard_write`,
which refuse any destination outside ``v4_grounded_next/``.  Sealed V4 inputs
are opened by sha256-verified read only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
VG = ROOT / "v4_grounded_next"
SEALED = ROOT / "refactor_v2"
DATA = ROOT / ".evaluation_runtime/data"
CV = DATA / "fracatlas_component_cv_v3"
MANIFEST = CV / "manifest.json"
FRACATLAS = DATA / "fracatlas/FracAtlas"
V4_REPORT = SEALED / "reports/local_noaux_refinement_v1"
RUNS = VG / "runs"
CACHE = VG / "cache"
CANONICAL_Q = "Is there a fracture in this X-ray?"
ROLES = ("train", "model_val", "calibration", "test")


def guard_write(path) -> Path:
    p = Path(path).resolve()
    if not str(p).startswith(str(VG.resolve()) + os.sep):
        raise PermissionError(f"Refusing to write outside v4_grounded_next/: {p}")
    if str(p).startswith(str(SEALED.resolve()) + os.sep):
        raise PermissionError(f"Refusing to write into sealed refactor_v2/: {p}")
    return p


def seed_dir(tag, fold) -> Path:
    """runs/arms/<tag>/fold_<f>/seed_<s>: seed_42 when it exists, else the only seed folder of the run."""
    base = RUNS / "arms" / tag / f"fold_{fold}"
    if (base / "seed_42").exists():
        return base / "seed_42"
    found = sorted(base.glob("seed_*"))
    assert len(found) == 1, f"{base}: expected one seed folder, found {[p.name for p in found]}"
    return found[0]


def sha256(path) -> str:
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def dump_json(path, value, exclusive: bool = False):
    p = guard_write(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with p.open("x") as f:
            json.dump(value, f, indent=2)
    else:
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(value, indent=2))
        tmp.replace(p)
    return p


def manifest_fold(fold) -> dict:
    """sha-verified role paths for an outer fold (from the sealed manifest)."""
    m = read_json(MANIFEST)
    spec = m["folds"][str(fold)]
    out = {}
    for role in ROLES:
        p = CV / f"fold_{fold}" / f"vqa_{role}.json"
        assert sha256(p) == spec[role]["sha256"], f"manifest sha mismatch {p}"
        out[role] = p
    return out


def component_lookup() -> dict:
    lookup = {}
    for gid, images in enumerate(read_json(MANIFEST)["components"]):
        for iid in images:
            assert iid not in lookup, iid
            lookup[iid] = gid
    return lookup


def image_table(records) -> dict:
    """image_id -> first record (image-level fields: GT boxes/polygons, size)."""
    table = {}
    for r in records:
        table.setdefault(r["image_id"], r)
    return table
