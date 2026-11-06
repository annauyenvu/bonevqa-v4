from types import SimpleNamespace

import pytest

from src.protocol import (assert_disjoint_splits, group_subsample, records_of,
                          validate_prompt_protocol_config)


def dataset(records):
    return SimpleNamespace(records=records, __len__=lambda self: len(records))


class Records:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def test_split_leakage_fails_closed():
    train = Records([{"image_id": "same", "answer": "yes"}])
    val = Records([{"image_id": "same", "answer": "no"}])
    with pytest.raises(RuntimeError, match="Leakage"):
        assert_disjoint_splits(train=train, val=val)


def test_group_subsample_never_splits_an_image():
    rows = [
        {"image_id": image, "answer": "yes", "row": row}
        for image in ("a", "b", "c") for row in range(3)
    ]
    subset = group_subsample(Records(rows), max_rows=5, seed=7)
    chosen = records_of(subset)
    counts = {image: sum(r["image_id"] == image for r in chosen) for image in ("a", "b", "c")}
    assert all(count in (0, 3) for count in counts.values())


def _v3_config(variant="B"):
    active = set() if variant == "B" else set(variant)
    return {
        "model": {
            "prompt_protocol_version": 3,
            "prompt_variant": variant,
            "use_visual_prompt": "V" in active,
            "use_lens": "L" in active,
            "use_latent_prompt": "P" in active,
            "use_textual_prompt": "T" in active,
            "use_region_in_prompt": False,
            "lens_training_only": True,
            "lens_dropout": 0.0,
            "qformer_dropout": 0.0,
            "visual_route": "fusion_only",
            "latent_answer_bank_mode": "train",
            "prior_labels": [],
        },
        "data": {"loader_kwargs": {
            "with_masks": False,
            "with_boxes": False,
            "boxes_from": "detector.json",
            "boxes_as_masks": True,
        }},
    }


def test_v3_baseline_is_truly_prompt_free():
    cfg = _v3_config("B")
    validate_prompt_protocol_config(cfg)
    cfg["model"]["use_textual_prompt"] = True
    with pytest.raises(RuntimeError, match="prompt_variant"):
        validate_prompt_protocol_config(cfg)


def test_v3_rejects_annotation_roi_and_fake_lapa_prior():
    cfg = _v3_config("V")
    cfg["data"]["loader_kwargs"]["with_masks"] = True
    with pytest.raises(RuntimeError, match="annotation-derived ROI"):
        validate_prompt_protocol_config(cfg)

    cfg = _v3_config("P")
    cfg["model"]["prior_labels"] = ["fracture", "normal"]
    with pytest.raises(RuntimeError, match="not LaPA prior fusion"):
        validate_prompt_protocol_config(cfg)

