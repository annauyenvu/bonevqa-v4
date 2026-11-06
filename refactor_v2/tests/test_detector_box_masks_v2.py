import json

import numpy as np
import pytest
from PIL import Image

from src.data.fracatlas_vqa import FracAtlasVQADataset


def test_detector_boxes_become_prompt_masks_without_annotation(tmp_path):
    boxes_path = tmp_path / "boxes.json"
    boxes_path.write_text(json.dumps({"img-1": [[1, 2, 5, 7]]}), encoding="utf-8")
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps([{
        "image_id": "img-1",
        "image": "unused.png",
        "question": "Is there a fracture?",
        "answer": "yes",
        "answer_type": "CLOSED",
        "boxes": [[0, 0, 9, 9]],
        "segmentation": [[[0, 0, 9, 0, 9, 9]]],
    }]), encoding="utf-8")
    dataset = FracAtlasVQADataset(
        records_path=records_path,
        root=tmp_path,
        with_masks=False,
        with_boxes=False,
        boxes_from=boxes_path,
        boxes_as_masks=True,
    )
    masks, boxes = dataset.build_masks(dataset.records[0], Image.new("RGB", (10, 10)))
    assert boxes == [[1, 2, 5, 7]]
    assert masks.shape == (1, 10, 10)
    assert int(masks.sum()) == 20
    assert np.all(masks[0, 2:7, 1:5] == 1)


def test_missing_detector_box_yields_empty_mask(tmp_path):
    boxes_path = tmp_path / "boxes.json"
    boxes_path.write_text("{}", encoding="utf-8")
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps([{
        "image_id": "img-2", "image": "unused.png", "question": "Q", "answer": "no"
    }]), encoding="utf-8")
    dataset = FracAtlasVQADataset(
        records_path=records_path,
        root=tmp_path,
        boxes_from=boxes_path,
        boxes_as_masks=True,
    )
    masks, boxes = dataset.build_masks(dataset.records[0], Image.new("RGB", (8, 6)))
    assert boxes is None
    assert masks.shape == (0, 6, 8)


def _controlled_dataset(tmp_path, roi_control):
    boxes_path = tmp_path / "controlled_boxes.json"
    boxes_path.write_text(json.dumps({
        "img-a": [[1, 2, 5, 7, 0.91]],
        "img-b": [[20, 0, 40, 10, 0.73]],
    }), encoding="utf-8")
    records_path = tmp_path / "controlled_records.json"
    records_path.write_text(json.dumps([
        {
            "image_id": "img-a", "image": "unused-a.png", "question": "Q",
            "answer": "yes", "width": 20, "height": 10,
        },
        {
            "image_id": "img-b", "image": "unused-b.png", "question": "Q",
            "answer": "no", "width": 40, "height": 20,
        },
    ]), encoding="utf-8")
    return FracAtlasVQADataset(
        records_path=records_path,
        root=tmp_path,
        boxes_from=boxes_path,
        boxes_as_masks=True,
        roi_control=roi_control,
        roi_seed=7,
    )


def test_shuffled_control_uses_detector_donor_and_changes_mask(tmp_path):
    image = Image.new("RGB", (20, 10))
    real = _controlled_dataset(tmp_path, "real")
    shuffled = _controlled_dataset(tmp_path, "shuffled")

    real_masks, _ = real.build_masks(real.records[0], image)
    masks, boxes = shuffled.build_masks(shuffled.records[0], image)

    # With two unique images the derangement must swap img-a and img-b.
    # The donor box is mapped by normalized coordinates (40x20 -> 20x10).
    assert boxes == [[10.0, 0.0, 20.0, 5.0, 0.73]]
    assert int(masks.sum()) == 50
    assert np.all(masks[0, 0:5, 10:20] == 1)
    assert not np.array_equal(masks, real_masks)


def test_generic_control_overrides_detector_mask(tmp_path):
    image = Image.new("RGB", (20, 10))
    real = _controlled_dataset(tmp_path, "real")
    generic = _controlled_dataset(tmp_path, "generic")

    real_masks, _ = real.build_masks(real.records[0], image)
    masks, boxes = generic.build_masks(generic.records[0], image)

    assert boxes == [[5, 2, 15, 7]]
    assert int(masks.sum()) == 50
    assert np.all(masks[0, 2:7, 5:15] == 1)
    assert not np.array_equal(masks, real_masks)


def test_matched_control_moves_detector_mask_but_preserves_geometry(tmp_path):
    image = Image.new("RGB", (20, 10))
    real = _controlled_dataset(tmp_path, "real")
    matched = _controlled_dataset(tmp_path, "matched")

    real_masks, real_boxes = real.build_masks(real.records[0], image)
    masks, boxes = matched.build_masks(matched.records[0], image)

    assert boxes != real_boxes
    assert len(boxes) == len(real_boxes) == 1
    assert boxes[0][2] - boxes[0][0] == real_boxes[0][2] - real_boxes[0][0]
    assert boxes[0][3] - boxes[0][1] == real_boxes[0][3] - real_boxes[0][1]
    assert boxes[0][4:] == real_boxes[0][4:]
    assert int(masks.sum()) == int(real_masks.sum()) == 20
    assert not np.array_equal(masks, real_masks)


def test_matched_control_does_not_invent_detector_roi(tmp_path):
    boxes_path = tmp_path / "boxes.json"
    boxes_path.write_text("{}", encoding="utf-8")
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps([{
        "image_id": "img-empty", "image": "unused.png", "question": "Q", "answer": "no"
    }]), encoding="utf-8")
    dataset = FracAtlasVQADataset(
        records_path=records_path,
        root=tmp_path,
        boxes_from=boxes_path,
        boxes_as_masks=True,
        roi_control="matched",
    )

    masks, boxes = dataset.build_masks(dataset.records[0], Image.new("RGB", (8, 6)))
    assert boxes is None
    assert masks.shape == (0, 6, 8)


def test_matched_control_fails_closed_without_detector_proposals(tmp_path):
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps([{
        "image_id": "img-1", "image": "unused.png", "question": "Q", "answer": "no"
    }]), encoding="utf-8")

    with pytest.raises(ValueError, match="matched.*boxes_from"):
        FracAtlasVQADataset(
            records_path=records_path,
            root=tmp_path,
            roi_control="matched",
        )


def test_matched_all_adds_geometry_matched_placebo_to_detector_negative(tmp_path):
    boxes_path = tmp_path / "boxes.json"
    boxes_path.write_text(json.dumps({"img-pos": [[2, 2, 6, 5]], "img-neg": []}), encoding="utf-8")
    records = [
        {"image": "pos.png", "image_id": "img-pos", "question": "q", "answer": "yes",
         "answer_type": "CLOSED", "width": 10, "height": 8},
        {"image": "neg.png", "image_id": "img-neg", "question": "q", "answer": "no",
         "answer_type": "CLOSED", "width": 10, "height": 8},
    ]
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")
    dataset = FracAtlasVQADataset(
        records_path=records_path, root=tmp_path, with_masks=False, with_boxes=False,
        boxes_from=boxes_path, boxes_as_masks=True, roi_control="matched_all",
    )
    masks, boxes = dataset.build_masks(records[1], Image.new("RGB", (10, 8)))
    assert len(boxes) == len(masks) == 1
    assert boxes[0][2] - boxes[0][0] == 4
    assert boxes[0][3] - boxes[0][1] == 3
