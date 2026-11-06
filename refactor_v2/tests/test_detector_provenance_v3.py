import hashlib
import json

import pytest

from src.preprocessing import RASTER_PREPROCESSING_VERSION
from src.provenance import validate_detector_proposals


def test_detector_provenance_round_trip(tmp_path):
    checkpoint = tmp_path / "detector.pt"
    split = tmp_path / "split.json"
    proposals = tmp_path / "boxes.json"
    checkpoint.write_bytes(b"checkpoint")
    split.write_text(json.dumps({"train": ["a"], "val": ["b"], "test": []}), encoding="utf-8")
    proposals.write_text(json.dumps({"a": [], "b": [[1, 2, 3, 4]]}), encoding="utf-8")
    artifact = {
        "schema_version": 3,
        "coordinate_space": "exif_canonical_pixels",
        "preprocessing_version": RASTER_PREPROCESSING_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "split_manifest": str(split),
        "split_manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "proposal_sha256": hashlib.sha256(proposals.read_bytes()).hexdigest(),
        "coverage": 2,
        "expected_coverage": 2,
        "complete": True,
    }
    proposals.with_suffix(".provenance.json").write_text(
        json.dumps({"artifact": artifact, "images": {"a": {}, "b": {}}}), encoding="utf-8"
    )
    assert validate_detector_proposals(proposals)["coverage"] == 2


def test_detector_provenance_rejects_modified_proposals(tmp_path):
    proposals = tmp_path / "boxes.json"
    proposals.write_text("{}", encoding="utf-8")
    proposals.with_suffix(".provenance.json").write_text(
        json.dumps({"artifact": {"schema_version": 3}}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError):
        validate_detector_proposals(proposals)
