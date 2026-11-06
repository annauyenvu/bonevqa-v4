#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "mandatory_experiments_20261011"
P = ROOT / "protocol"
LEGACY = PROJECT / ".evaluation_runtime"
SPLIT = LEGACY / "data" / "fracatlas_component_clean"
SEEDS = [42, 123, 456, 789, 2026]
VARIANTS = ["none", "v", "p", "vp"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    if (P / "LOCK.json").exists():
        raise RuntimeError("protocol already sealed")
    P.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for split, limit in (("train", 6000), ("val", 300), ("test", None)):
        rows = json.loads((SPLIT / f"vqa_{split}.json").read_text())
        manifests[split] = rows[:limit]
        write(P / f"{split}_manifest.json", manifests[split])
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not ({row["image_id"] for row in manifests[left]} & {row["image_id"] for row in manifests[right]})
    specs = {
        split: [len(rows), len({row["image_id"] for row in rows}), sum(row["answer_type"] == "CLOSED" for row in rows)]
        for split, rows in manifests.items()
    }
    write(P / "specs.json", specs)

    base = json.loads((PROJECT / "exp4_multiseed/protocol/base_config.json").read_text())
    base["run_name"] = "mandatory_detector_factorial_component_clean"
    base["seed"] = None
    base["checkpoint_dir"] = str(ROOT)
    base["data"]["root"] = str(LEGACY / "data")
    base["data"]["dataset"] = "fracatlas_component_clean"
    base["data"]["max_samples"] = 6000
    base["data"]["val_max_samples"] = 300
    base["data"]["num_workers"] = 0
    base["train"]["epochs"] = 2
    base["train"]["batch_size"] = 4
    base["train"]["grad_accum"] = 4
    base["train"]["amp"] = False
    base["model"]["llm_dtype"] = "float32"
    base["model"]["sam"]["sam_med2d_ckpt"] = None
    base["model"]["sam"]["cache_dir"] = str(ROOT / "unused_sam_cache")
    base["init_from"] = str(LEGACY / "checkpoints/stage2_pmcvqa_full/best.pt")
    write(P / "base_config.json", base)
    answer_vocab = json.loads((PROJECT / "exp4_multiseed/protocol/answer_vocab.json").read_text())
    write(P / "answer_vocab.json", answer_vocab)

    images = {}
    for rows in manifests.values():
        for row in rows:
            images.setdefault(row["image_id"], row)
    cohort = []
    fracatlas = LEGACY / "data" / "fracatlas" / "FracAtlas"
    for image_id, row in sorted(images.items()):
        path = fracatlas / row["image"]
        cohort.append({"image_id": image_id, "path": str(path), "sha256": sha(path),
                       "width": row["width"], "height": row["height"]})
    write(P / "image_content_manifest.json", cohort)

    init = Path(base["init_from"])
    detector = LEGACY / "data" / "fracatlas" / "box_bo_phat_hien.json"
    protocol = {
        "objective": "five-seed Detector V-vs-None and None/V/P/VP interaction on conservative component-clean split",
        "seeds": SEEDS,
        "variants": VARIANTS,
        "primary_runs": len(SEEDS) * len(VARIANTS),
        "primary_contrasts": ["V - None", "(VP - P) - (V - None)"],
        "split": "components connect exact duplicates and all-pair perceptual near-duplicate candidates",
        "training": "2 epochs; 6000 train QA; 300 validation QA; batch 4; grad accumulation 4; float32; MPS deterministic",
        "checkpoint_selection": "validation overall exact accuracy; ties retain earliest epoch",
        "primary_reporting": ["closed exact", "overall exact", "open exact", "macro family accuracy"],
        "no_test_selection": True,
        "specs": specs,
    }
    write(P / "protocol.json", protocol)
    inputs = {
        str(init): sha(init),
        str(detector): sha(detector),
        str(SPLIT / "component_split_manifest.json"): sha(SPLIT / "component_split_manifest.json"),
        str(LEGACY / "provenance_audit/exact_duplicate_manifest.json"): sha(LEGACY / "provenance_audit/exact_duplicate_manifest.json"),
        str(LEGACY / "provenance_audit/near_duplicates_all/near_duplicate_candidates.json"): sha(LEGACY / "provenance_audit/near_duplicates_all/near_duplicate_candidates.json"),
    }
    write(P / "input_hashes.json", inputs)

    # Check that the stage-2 payload can initialize the locked vocabulary/model.
    payload = torch.load(init, map_location="cpu", weights_only=False)
    write(P / "initialization_audit.json", {"checkpoint_sha256": inputs[str(init)],
                                             "payload_keys": sorted(payload),
                                             "stage2_answer_vocab_size": len(payload["answer_vocab"]),
                                             "downstream_answer_vocab_size": len(answer_vocab),
                                             "answer_vocab_identity": payload["answer_vocab"] == answer_vocab,
                                             "note": "downstream vocabulary is locked from the historical FracAtlas checkpoint; shape-compatible stage-2 weights are loaded"})
    print(json.dumps({"specs": specs, "unique_images": len(cohort), "runs": protocol["primary_runs"]}, indent=2))


if __name__ == "__main__":
    main()
