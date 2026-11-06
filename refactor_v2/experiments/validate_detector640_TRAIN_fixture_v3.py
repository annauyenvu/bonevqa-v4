"""Strong scoped TRAIN-fixture contract;never authorize a scientific CLI."""
import copy
import hashlib
import json
from pathlib import Path

import torch
from src.provenance import validate_detector_proposals
from experiments.audit_detector640_top3_binding_v3 import scoped_check,sha


def check_metadata(proposals,side):
    artifact=side["artifact"]
    checkpoint=torch.load(artifact["checkpoint"],map_location="cpu",weights_only=False)
    assert artifact["detector_transform"]==checkpoint["detector_transform"]
    norm_hash=hashlib.sha256(json.dumps(checkpoint["normalization_spec"],sort_keys=True).encode()).hexdigest()
    assert artifact["normalization_spec_sha256"]==norm_hash
    assert checkpoint["canh_dai"]==640
    gate=json.loads(Path(artifact["TRAIN_runtime_gate"]).read_text())
    assert gate["passed"] and gate["scientific_proposal_or_VLP_training_released"] is False
    assert gate["provenance"]["checkpoint_sha256"]==artifact["checkpoint_sha256"]
    assert gate["provenance"]["detector_transform"]==artifact["detector_transform"]
    assert gate["provenance"]["normalization_spec_sha256"]==norm_hash
    expected={r["TRAIN_image_id"]:r["boxes"] for r in gate["rows"]}
    assert proposals==expected and set(side["images"])==set(expected)
    for iid,boxes in proposals.items():
        row=side["images"][iid]
        assert row["split"]=="train" and len(boxes)<=3
        assert all(0<=b[0]<b[2]<=row["width"] and 0<=b[1]<b[3]<=row["height"] for b in boxes)
    return True


def validate(path):
    path=Path(path)
    scoped_check(path)
    proposals=json.loads(path.read_text())
    side=json.loads(path.with_suffix(".provenance.json").read_text())
    check_metadata(proposals,side)
    return side["artifact"]


def main():
    root=Path.cwd()
    folder=root/"refactor_v2/reports/detector640_top3_binding_v3"
    output=folder/"strong_contract_verification.json"
    assert not output.exists()
    path=folder/"TRAIN_technical_boxes.json"
    validate(path)
    proposals=json.loads(path.read_text())
    side=json.loads(path.with_suffix(".provenance.json").read_text())
    rejected={}
    for fault in ("normalization_hash","claimed_validation_role","proposal_coordinates"):
        altered=copy.deepcopy(side)
        boxes=copy.deepcopy(proposals)
        if fault=="normalization_hash":
            altered["artifact"]["normalization_spec_sha256"]="0"*64
        elif fault=="claimed_validation_role":
            altered["images"][next(iter(boxes))]["split"]="val"
        else:
            key=next(k for k,v in boxes.items() if v)
            boxes[key][0][0]+=1
        try:
            check_metadata(boxes,altered)
        except AssertionError:
            rejected[fault]=True
        else:
            raise RuntimeError("Fault passed:"+fault)
    result={"passed":True,"scope":"exact8TRAINfixtureonly;not scientific data/proposal release","proposal_sha256":sha(path),"sidecar_sha256":sha(path.with_suffix(".provenance.json")),"validator_source_sha256":sha(__file__),"checkpoint_normalization_transform_and_fixture_boxes_exact":True,"negative_metadata_cases_rejected":rejected,"complete_production_guard_unchanged":True,"test_or_calibration_used":False,"scientific_VLP_or_production_released":False}
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
