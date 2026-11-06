"""Bind the sealed EXP4 engine to the mandatory clean-split experiment root."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "mandatory_experiments_20261011"
PROTOCOL = ROOT / "protocol"
EXP4_CODE = PROJECT / "exp4_multiseed" / "code"
LEGACY = PROJECT / ".evaluation_runtime"


def configure():
    sys.path[:0] = [str(EXP4_CODE), str(LEGACY / "deps")]
    import common
    import engine

    specs_raw = json.loads((PROTOCOL / "specs.json").read_text())
    specs = {name: tuple(values) for name, values in specs_raw.items()}
    for module in (common, engine):
        module.ROOT = ROOT
        module.P = PROTOCOL
        module.LEGACY = LEGACY
        module.CODE = ROOT / "code"
        module.SPECS = specs
        module.SEEDS = [42, 123, 456, 789, 2026]
        module.REGIMES = ["detector"]
        module.CONFIGS = ["none", "v", "p", "vp"]

    os.environ["MPLCONFIGDIR"] = str(ROOT / "mpl")
    os.chdir(ROOT)

    def run_config(regime: str, variant: str, seed: int) -> dict:
        assert regime == "detector"
        assert variant in {"none", "v", "p", "vp"}
        cfg = common.read(PROTOCOL / "base_config.json")
        cfg["seed"] = seed
        cfg["localization_regime"] = regime
        cfg["vlp_switch"] = variant
        cfg["model"]["use_visual_prompt"] = "v" in variant
        cfg["model"]["use_lens"] = False
        cfg["model"]["use_latent_prompt"] = "p" in variant
        return cfg

    def finish_roi_audit(dest, regime, variant, seen, complete=True):
        source = common.read(ROOT / "roi_bank" / regime / "audit.json")
        source["roi_branch_enabled"] = "v" in variant
        source["run_dataset_access"] = {
            split: {"expected_images": specs[split][1], "processed_images": len(seen.get(split, set()))}
            for split in specs
        }
        source["expected_images"] = len({row["image_id"] for split in specs for row in common.read(PROTOCOL / f"{split}_manifest.json")})
        source["processed_images"] = len(set().union(*seen.values())) if seen else 0
        source["roi_success"] = source["processed_images"]
        source["roi_failed"] = 0
        source["bank_index_sha256"] = common.sha(ROOT / "roi_bank" / regime / "index.json")
        source["complete"] = complete
        if complete:
            assert source["processed_images"] == source["expected_images"]
            for split in specs:
                assert len(seen.get(split, set())) == specs[split][1]
        common.write(Path(dest) / "roi_audit.json", source)

    original_dataset = engine.FrozenDataset

    class ComponentCleanDataset(original_dataset):
        """EXP4 frozen dataset with EXIF orientation aligned to metadata/boxes."""

        def __getitem__(self, index):
            row = self.records[index]
            image_id = row["image_id"]
            roi = self.index[image_id]
            image_meta = self.images[image_id]
            if image_id not in self.checked:
                assert common.sha(image_meta["path"]) == image_meta["sha256"], "Image content changed"
                assert common.sha(roi["path"]) == roi["mask_sha256"], "ROI corrupted"
                assert roi["roi_failed"] == 0 and roi["roi_success"]
                self.checked.add(image_id)
            with Image.open(image_meta["path"]) as source:
                expected_size = (image_meta["width"], image_meta["height"])
                image = (source if source.size == expected_size else ImageOps.exif_transpose(source)).convert("RGB")
            assert image.size == (image_meta["width"], image_meta["height"])
            with np.load(roi["path"]) as data:
                masks = data["masks"].copy()
            assert masks.shape == (roi["mask_count"], image.height, image.width)
            self.audit.setdefault(self.split, set()).add(image_id)
            return {"image": image, "question": row["question"], "answer": row["answer"],
                    "answer_type": row["answer_type"], "region": row.get("region"),
                    "image_id": image_id, "masks": masks, "boxes": None}

    engine.run_config = run_config
    engine.finish_roi_audit = finish_roi_audit
    engine.FrozenDataset = ComponentCleanDataset
    return common, engine, specs
