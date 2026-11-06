"""Fail-closed provenance checks for detector proposal artifacts."""

from __future__ import annotations

import hashlib
import json
import torch
from pathlib import Path

from .preprocessing import RASTER_PREPROCESSING_VERSION
from .detector_protocol import load_detector_manifest, verify_detector_checkpoint_binding


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_detector_proposals(path: str | Path, require_complete: bool = True) -> dict:
    proposal_path = Path(path)
    provenance_path = proposal_path.with_suffix(".provenance.json")
    if not proposal_path.exists() or not provenance_path.exists():
        raise RuntimeError("Thiếu detector proposals hoặc provenance sidecar")
    proposals = json.loads(proposal_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    artifact = provenance.get("artifact") or {}
    if int(artifact.get("schema_version", 0)) < 3:
        raise RuntimeError("Detector proposal provenance schema phải >= 3")
    if artifact.get("coordinate_space") != "exif_canonical_pixels":
        raise RuntimeError("Detector proposals không ở EXIF-canonical coordinate space")
    if artifact.get("preprocessing_version") != RASTER_PREPROCESSING_VERSION:
        raise RuntimeError("Detector proposal preprocessing version không khớp runtime")
    if artifact.get("proposal_sha256") != _sha256(proposal_path):
        raise RuntimeError("Detector proposal SHA-256 mismatch")
    if int(artifact.get("coverage", -1)) != len(proposals):
        raise RuntimeError("Detector proposal coverage không khớp payload")
    sources = {}
    for path_key, hash_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("split_manifest", "split_manifest_sha256"),
    ):
        source = Path(str(artifact.get(path_key, "")))
        if not source.is_file() or artifact.get(hash_key) != _sha256(source):
            raise RuntimeError(f"Detector provenance source mismatch: {path_key}")
        sources[path_key] = source
    split_payload, split_ids, split_sha = load_detector_manifest(sources["split_manifest"])
    if split_payload.get("protocol_version", 0) >= 3:
        checkpoint = torch.load(sources["checkpoint"], map_location="cpu", weights_only=False)
        verify_detector_checkpoint_binding(checkpoint, split_payload, split_sha)
        if set(checkpoint.get("training_image_ids", [])) != set(split_ids["train"]):
            raise RuntimeError("Detector checkpoint actual training IDs mismatch")
        if require_complete and split_payload.get("scope") == "technical_smoke_train_only":
            raise RuntimeError("Technical-smoke detector proposals are not scientific fold inputs")
    expected_ids = set().union(*(set(ids) for ids in split_ids.values()))
    if int(artifact.get("expected_coverage", -1)) != len(expected_ids):
        raise RuntimeError("Detector expected coverage không khớp split manifest")
    if require_complete and (artifact.get("complete") is not True or set(proposals) != expected_ids):
        raise RuntimeError("Detector proposal artifact chưa phủ toàn bộ sealed splits")
    images = provenance.get("images") or {}
    if set(images) != set(proposals):
        raise RuntimeError("Detector proposal per-image provenance không phủ đúng payload")
    return artifact
