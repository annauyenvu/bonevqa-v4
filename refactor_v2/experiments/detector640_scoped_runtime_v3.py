"""Experimental image-only640adapter;never replace sealed production runtime."""
import hashlib
import json
from pathlib import Path

import torch
from src.detector_runtime import FractureDetectorRuntime
from src.detector_protocol import load_detector_manifest,verify_detector_checkpoint_binding
from src.preprocessing import RASTER_PREPROCESSING_VERSION
from experiments.detector640_checkpoint_v3 import load_bound_640


class ScopedDetector640(FractureDetectorRuntime):
    def __init__(self,checkpoint,expected_sha256,manifest_path,*,technical_only):
        if technical_only is not True:
            raise RuntimeError("No production or scientific proposal release from technical adapter")
        self.checkpoint=Path(checkpoint)
        digest=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        assert digest==expected_sha256
        self.model,payload=load_bound_640(checkpoint)
        manifest,_,split_hash=load_detector_manifest(manifest_path)
        verify_detector_checkpoint_binding(payload,manifest,split_hash)
        assert payload["preprocessing_version"]==RASTER_PREPROCESSING_VERSION
        self.device=torch.device("cpu")
        self.image_size=640
        self.score_threshold=0.05
        self.max_boxes=3
        self.model.eval()
        self.provenance={"checkpoint":str(self.checkpoint),"checkpoint_sha256":digest,"split_manifest_sha256":split_hash,"preprocessing_version":RASTER_PREPROCESSING_VERSION,"architecture":payload["architecture"],"detector_transform":payload["detector_transform"],"normalization_spec_sha256":hashlib.sha256(json.dumps(payload["normalization_spec"],sort_keys=True).encode()).hexdigest(),"score_threshold":0.05,"max_boxes":3,"scope":"TRAIN-only runtime gate;NOTproduction or scientific proposal export"}
