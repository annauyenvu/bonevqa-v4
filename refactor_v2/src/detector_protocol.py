"""Detector split/checkpoint binding; never relabel old weights as a new fold."""
import hashlib
import itertools
import json
from pathlib import Path

from .data.sealed_manifest import load_sealed_records


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_detector_manifest(path):
    path = Path(path)
    payload = json.loads(path.read_text())
    splits = payload.get("split_images", payload)
    roles = [role for role in ("train", "val", "calibration", "test") if role in splits]
    for required in ("train", "val", "test"):
        if required not in roles or (payload.get("protocol_version", 0) >= 3 and not splits[required]):
            raise RuntimeError("Missing/empty detector split: " + required)
    for role in roles:
        if len(splits[role]) != len(set(splits[role])):
            raise RuntimeError("Duplicate detector image IDs within " + role)
    for left, right in itertools.combinations(roles, 2):
        if set(splits[left]) & set(splits[right]):
            raise RuntimeError("Detector split leakage: " + left + "/" + right)
    if payload.get("protocol_version", 0) >= 3:
        seal = payload["source_seal"]
        if file_sha256(seal["path"]) != seal["sha256"]:
            raise RuntimeError("Detector source data seal hash mismatch")
        for role in roles:
            source = payload["source_vqa"][role]
            if file_sha256(source["path"]) != source["sha256"]:
                raise RuntimeError("Detector source VQA hash mismatch")
            rows = load_sealed_records(source["path"], seal["path"], payload["project_root"])
            expected = {row["image_id"] for row in rows}
            actual = set(splits[role])
            smoke = payload.get("scope") == "technical_smoke_train_only"
            if (smoke and not actual <= expected) or (not smoke and actual != expected):
                raise RuntimeError("Detector/VQA partition parity failed: " + role)
        components = payload["component_by_image"]
        seen = {}
        for role in roles:
            for image in splits[role]:
                component = components[image]
                if component in seen and seen[component] != role:
                    raise RuntimeError("Detector component crosses partition boundaries")
                seen[component] = role
    return payload, {role: list(splits[role]) for role in roles}, file_sha256(path)


def verify_detector_checkpoint_binding(checkpoint, manifest_payload, manifest_sha256):
    if manifest_payload.get("protocol_version", 0) >= 3:
        if checkpoint.get("training_split_sha256") != manifest_sha256:
            raise RuntimeError("Detector checkpoint was not bound to this fold/split manifest")
        if checkpoint.get("training_scope") != manifest_payload.get("scope"):
            raise RuntimeError("Detector checkpoint training scope mismatch")


def verify_detector_annotation_parity(dataset, source_vqa_path):
    expected = {}
    for row in json.loads(Path(source_vqa_path).read_text()):
        value = (row["image"], row.get("boxes") or [])
        if row["image_id"] in expected and expected[row["image_id"]] != value:
            raise RuntimeError("Conflicting source detector annotations")
        expected[row["image_id"]] = value
    for image_id, (relative_image, boxes) in dataset.muc:
        if expected.get(image_id) != (relative_image, boxes or []):
            raise RuntimeError("Detector annotation/cache differs from sealed VQA ground truth: " + image_id)
