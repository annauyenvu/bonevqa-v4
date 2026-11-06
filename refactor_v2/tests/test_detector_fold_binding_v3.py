import json

import pytest

from src.detector_protocol import file_sha256, load_detector_manifest, verify_detector_checkpoint_binding
from src.detector_protocol import verify_detector_annotation_parity


def test_old_detector_cannot_be_relabelled_as_new_fold():
    with pytest.raises(RuntimeError, match="not bound"):
        verify_detector_checkpoint_binding({}, {"protocol_version": 3, "scope": "fold"}, "newhash")
    with pytest.raises(RuntimeError, match="not bound"):
        verify_detector_checkpoint_binding({"training_split_sha256": "oldhash"},
                                           {"protocol_version": 3, "scope": "fold"}, "newhash")
    verify_detector_checkpoint_binding({"training_split_sha256": "newhash", "training_scope": "fold"},
                                       {"protocol_version": 3, "scope": "fold"}, "newhash")


def test_detector_calibration_cannot_overlap_train(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"split_images": {"train": ["a"], "val": ["b"],
                                                "calibration": ["a"], "test": ["c"]}}))
    with pytest.raises(RuntimeError, match="leakage"):
        load_detector_manifest(path)


def test_bound_checkpoint_cannot_change_training_scope():
    with pytest.raises(RuntimeError, match="scope mismatch"):
        verify_detector_checkpoint_binding({"training_split_sha256": "x", "training_scope": "smoke"},
                                           {"protocol_version": 3, "scope": "fold"}, "x")


def bound_manifest(tmp_path):
    roles = {"train": ["a"], "val": ["b"], "calibration": ["c"], "test": ["d"]}
    artifacts, sources = {}, {}
    for role, ids in roles.items():
        source = tmp_path / (role + ".json")
        source.write_text(json.dumps([{"image_id": image} for image in ids]))
        artifacts[source.name] = file_sha256(source)
        sources[role] = {"path": str(source), "sha256": file_sha256(source)}
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({"status": "sealed_development_data", "artifacts": artifacts}))
    payload = {"protocol_version": 3, "scope": "fold", "project_root": str(tmp_path),
               "split_images": roles, "source_vqa": sources,
               "source_seal": {"path": str(seal), "sha256": file_sha256(seal)},
               "component_by_image": {image: image for ids in roles.values() for image in ids}}
    path = tmp_path / "detector.json"
    path.write_text(json.dumps(payload))
    return path, payload


def test_detector_four_role_binding_and_tamper_rejection(tmp_path):
    path, payload = bound_manifest(tmp_path)
    _, splits, _ = load_detector_manifest(path)
    assert splits["calibration"] == ["c"]
    source = tmp_path / "train.json"
    source.write_text("[]")
    with pytest.raises(RuntimeError, match="source VQA hash"):
        load_detector_manifest(path)


def test_detector_component_overlap_is_rejected(tmp_path):
    path, payload = bound_manifest(tmp_path)
    payload["component_by_image"]["d"] = "a"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="component crosses"):
        load_detector_manifest(path)


def test_detector_annotation_cache_must_match_sealed_gt(tmp_path):
    from types import SimpleNamespace
    path = tmp_path / "vqa.json"
    path.write_text(json.dumps([{"image_id": "a", "image": "a.jpg", "boxes": [[1, 2, 3, 4]]}]))
    verify_detector_annotation_parity(SimpleNamespace(muc=[("a", ("a.jpg", [[1, 2, 3, 4]]))]), path)
    with pytest.raises(RuntimeError, match="differs from sealed"):
        verify_detector_annotation_parity(SimpleNamespace(muc=[("a", ("a.jpg", []))]), path)
