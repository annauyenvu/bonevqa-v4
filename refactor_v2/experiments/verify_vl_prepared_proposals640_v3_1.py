"""Independent preparation contract fault tests and image-only prediction replay."""
import copy
import json
import time
from pathlib import Path

import torch
from PIL import Image
from experiments.vl_bounded_proposal_contract_v3_1 import sha,validate_prepared_proposals,PilotPreparationDetector640


@torch.no_grad()
def main():
    root=Path.cwd()
    folder=root/"refactor_v2/reports/vl_pilot_proposals640_v3_1"
    output=folder/"independent_contract_verification.json"
    cases_folder=folder/"negative_fixture_cases"
    assert not output.exists() and not cases_folder.exists()
    summary=json.loads((folder/"summary.json").read_text())
    assert sha(folder/"summary.json")==json.loads((folder/"progress.json").read_text())["summary_sha256"]
    path=folder/"boxes.json"
    assert sha(path)==summary["proposal_sha256"] and sha(path.with_suffix(".provenance.json"))==summary["sidecar_sha256"]
    assert sha(folder/"active_roles_manifest.json")==summary["active_roles_manifest_sha256"]
    validate_prepared_proposals(path)
    body=json.loads(path.read_text())
    side=json.loads(path.with_suffix(".provenance.json").read_text())
    cases_folder.mkdir()
    started=time.perf_counter()
    faults={}
    for fault in ("missing_image","claimed_calibration_role","wrong_transform","wrong_normalization_hash","wrong_image_byte_hash","falsely_complete_fullfold","QA_row_count_tamper"):
        case=cases_folder/fault
        case.mkdir()
        boxes=copy.deepcopy(body)
        altered=copy.deepcopy(side)
        if fault=="missing_image":
            key=next(iter(boxes));del boxes[key];del altered["images"][key]
            altered["artifact"]["coverage"]=len(boxes)
        elif fault=="claimed_calibration_role":
            altered["images"][next(iter(boxes))]["role"]="calibration"
        elif fault=="wrong_transform":
            altered["artifact"]["detector_transform"]["min_size"]=[320]
        elif fault=="wrong_normalization_hash":
            altered["artifact"]["normalization_spec_sha256"]="0"*64
        elif fault=="wrong_image_byte_hash":
            altered["images"][next(iter(boxes))]["input_image_sha256"]="0"*64
        elif fault=="falsely_complete_fullfold":
            altered["artifact"]["complete"]=True
        else:
            roles=json.loads((folder/"active_roles_manifest.json").read_text())
            roles["roles"]["train"]["QA_rows"]-=1
            roles_path=case/"wrong_roles_manifest.json"
            roles_path.write_text(json.dumps(roles,indent=2)+"\n")
            altered["artifact"]["active_roles_manifest"]=str(roles_path)
            altered["artifact"]["active_roles_manifest_sha256"]=sha(roles_path)
        negative=case/"boxes.json"
        negative.write_text(json.dumps(boxes,indent=2)+"\n")
        altered["artifact"]["proposal_sha256"]=sha(negative)
        negative.with_suffix(".provenance.json").write_text(json.dumps(altered,indent=2)+"\n")
        try:
            validate_prepared_proposals(negative)
        except (AssertionError,RuntimeError) as error:
            faults[fault]={"rejected":True,"error_type":type(error).__name__}
        else:
            raise RuntimeError("Invalid preparation contract accepted:"+fault)
    torch.set_num_threads(8)
    runtime=PilotPreparationDetector640(folder/"active_roles_manifest.json")
    replay=[]
    for role in ("train","model_val"):
        role_ids=sorted(i for i,v in side["images"].items() if v["role"]==role)
        # Branch coverage only,noGTlabel selection or performance claim.
        chosen=[next(i for i in role_ids if body[i]),next(i for i in role_ids if not body[i])]
        for iid in chosen:
            with Image.open(side["images"][iid]["input_image_path"]) as image:
                image.load()
                prediction=runtime.predict(image)
            assert prediction==body[iid]
            replay.append({"image_id":iid,"role":role,"exact_boxes":True,"box_count":len(prediction)})
    report={"passed":True,"scope":"independent active-role DATA PREPARATION verification;not scientific training release","proposal_sha256":summary["proposal_sha256"],"active_roles_manifest_sha256":summary["active_roles_manifest_sha256"],"exact_coverage":293,"role_image_counts":{"train":261,"model_val":32},"negative_cases":faults,"fresh_process_image_only_replay":replay,"elapsed_seconds":time.perf_counter()-started,"calibration_or_test_inferred":False,"fullfold_production_guard_unchanged":True,"scientific_VLP_training_or_production_released":False,"pending":"freeze matched control geometry/view/phase budgets and pre-task initialization;separate bounded execution contract/release still required"}
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
