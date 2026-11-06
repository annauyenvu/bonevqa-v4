"""Data-preparation role contract;never weaken complete-fold production guard."""
import hashlib
import json
from pathlib import Path

import torch
from src.data.sealed_manifest import load_sealed_records
from src.detector_protocol import load_detector_manifest,verify_detector_checkpoint_binding
from src.detector_runtime import FractureDetectorRuntime
from src.preprocessing import RASTER_PREPROCESSING_VERSION
from experiments.detector640_checkpoint_v3 import load_bound_640


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_roles(path):
    spec=json.loads(Path(path).read_text())
    assert spec["scope"]=="bounded_pilot_data_preparation_only" and spec["training_released"] is False
    assert set(spec["roles"])=={"train","model_val"}
    assert sha(spec["source_QA_seal"]["path"])==spec["source_QA_seal"]["sha256"]
    parent,foldids,foldsha=load_detector_manifest(spec["parent_detector_manifest"]["path"])
    assert foldsha==spec["parent_detector_manifest"]["sha256"]
    components=[]
    for role,source in spec["roles"].items():
        assert sha(source["path"])==source["sha256"]
        rows=load_sealed_records(source["path"],spec["source_QA_seal"]["path"],spec["project_root"])
        active=sorted({r["image_id"] for r in rows})
        assert active==source["image_ids"] and len(rows)==source["QA_rows"]
        assert set(active)<=set(foldids["train" if role=="train" else "val"])
        assert not set(active)&(set(foldids["calibration"])|set(foldids["test"]))
        components.append({parent["component_by_image"][iid] for iid in active})
    assert not components[0]&components[1]
    assert not set(spec["roles"]["train"]["image_ids"])&set(spec["roles"]["model_val"]["image_ids"])
    return spec


class PilotPreparationDetector640(FractureDetectorRuntime):
    def __init__(self,roles_path):
        spec=validate_roles(roles_path)
        source=spec["detector_checkpoint"]
        self.checkpoint=Path(source["path"])
        assert sha(self.checkpoint)==source["sha256"]
        self.model,payload=load_bound_640(self.checkpoint)
        manifest,_,digest=load_detector_manifest(spec["parent_detector_manifest"]["path"])
        verify_detector_checkpoint_binding(payload,manifest,digest)
        assert payload["epoch"]==2 and payload["optimizer_steps"]==1408
        assert payload["preprocessing_version"]==RASTER_PREPROCESSING_VERSION
        self.device=torch.device("cpu")
        self.image_size=640
        self.score_threshold=0.05
        self.max_boxes=3
        self.model.eval()
        self.provenance={"checkpoint":str(self.checkpoint),"checkpoint_sha256":source["sha256"],"detector_transform":payload["detector_transform"],"normalization_spec_sha256":hashlib.sha256(json.dumps(payload["normalization_spec"],sort_keys=True).encode()).hexdigest(),"scope":"activeTRAIN_modelvalidation_data_preparation_only,NOTtraining_or_production_release"}


def validate_prepared_proposals(path):
    from src.provenance import validate_detector_proposals
    path=Path(path)
    artifact=validate_detector_proposals(path,require_complete=False)
    assert artifact["scope"]=="bounded_pilot_data_preparation_only"
    assert artifact["complete"] is False and artifact["complete_for_active_roles"] is True
    assert artifact["training_released"] is False
    assert sha(artifact["active_roles_manifest"])==artifact["active_roles_manifest_sha256"]
    spec=validate_roles(artifact["active_roles_manifest"])
    boxes=json.loads(path.read_text())
    side=json.loads(path.with_suffix(".provenance.json").read_text())
    expected={iid:role for role,source in spec["roles"].items() for iid in source["image_ids"]}
    assert set(boxes)==set(side["images"])==set(expected)
    assert artifact["active_expected_coverage"]==len(expected)
    assert artifact["checkpoint_sha256"]==spec["detector_checkpoint"]["sha256"]
    checkpoint=torch.load(artifact["checkpoint"],map_location="cpu",weights_only=False)
    assert artifact["detector_transform"]==checkpoint["detector_transform"]
    assert artifact["normalization_spec_sha256"]==hashlib.sha256(json.dumps(checkpoint["normalization_spec"],sort_keys=True).encode()).hexdigest()
    assert artifact["score_threshold"]==0.05 and artifact["max_boxes"]==3
    for key in ("runtime_source","exporter_source"):
        assert sha(artifact[key])==artifact[key+"_sha256"]
    for iid,items in boxes.items():
        row=side["images"][iid]
        assert row["role"]==expected[iid] and sha(row["input_image_path"])==row["input_image_sha256"]
        assert len(items)<=3 and all(0<=b[0]<b[2]<=row["width"] and 0<=b[1]<b[3]<=row["height"] for b in items)
    return artifact
