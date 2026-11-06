"""Original-image TRAIN fixtures,raw640preprocessing and proposal coordinates."""
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image
from scripts.train_detector_v2 import BoPhatHienDataset
from src.data.fracatlas_vqa import FRACATLAS_ROOT
from experiments.detector640_scoped_runtime_v3 import ScopedDetector640


@torch.no_grad()
def main():
    started=time.perf_counter()
    root=Path.cwd()
    folder=root/"refactor_v2/reports/detector640_fold0_bounded_quality_v3_1"
    output=folder/"TRAIN_image_runtime_gate.json"
    assert not output.exists()
    done=json.loads((folder/"completion.json").read_text())
    release=json.loads((folder/"execution_release.json").read_text())
    evaluation=folder/"exact_architecture_independent_verification.json"
    assert hashlib.sha256(evaluation.read_bytes()).hexdigest()==json.loads((folder/"exact_architecture_verification_progress.json").read_text())["report_sha256"]
    assert json.loads(evaluation.read_text())["passed_artifact_and_exact_architecture_metric_checks"]
    manifest=json.loads(Path(release["manifest_path"]).read_text())
    ds=BoPhatHienDataset("train",canh_dai=640,tang_cuong=False,allowed_ids=set(manifest["split_images"]["train"]))
    chosen=[i for positive in (True,False) for i,_ in list((i,v) for i,v in enumerate(ds.muc) if bool(v[1][1])==positive)[:4]]
    assert len(chosen)==8
    torch.set_num_threads(8)
    runtime=ScopedDetector640(root/"refactor_v2/detector640_fold0_bounded_quality_v3_1/last.pt",done["last_sha256"],release["manifest_path"],technical_only=True)
    rows=[]
    for index in chosen:
        iid,(relative,_) = ds.muc[index]
        reference,_,_=ds[index]
        with Image.open(Path(FRACATLAS_ROOT)/relative) as raw:
            raw.load()
            canonical,tensor,scale,pad_x,pad_y=runtime._preprocess(raw)
            error=float((tensor-reference).abs().max())
            assert error==0
            prediction=runtime.model([tensor])[0]
            expected=[]
            for box,score in zip(prediction["boxes"].tolist(),prediction["scores"].tolist()):
                if score<0.05 or len(expected)>=3:
                    continue
                x1=max(0,min(canonical.width,(box[0]-pad_x)/scale));y1=max(0,min(canonical.height,(box[1]-pad_y)/scale))
                x2=max(0,min(canonical.width,(box[2]-pad_x)/scale));y2=max(0,min(canonical.height,(box[3]-pad_y)/scale))
                if x2>x1 and y2>y1:
                    expected.append([x1,y1,x2,y2])
            actual=runtime.predict(raw)
            assert actual==expected
            assert all(0<=b[0]<b[2]<=canonical.width and 0<=b[1]<b[3]<=canonical.height for b in actual)
            rows.append({"TRAIN_image_id":iid,"preprocessing_tensor_max_error":error,"number_automatic_boxes":len(actual),"boxes":actual,"canonical_image_size":list(canonical.size),"direct_prediction_coordinate_decode_exact":True})
    report={"passed":True,"scope":"8originalTRAINimage-onlyruntimefixtures;not modelperformance or production release","provenance":runtime.provenance,"rows":rows,"inference_input":"original image only;noGTboxes,answer,type orROI","GT_used_only_for_fixture_selection":True,"empty_proposals_preserved":True,"warning":"Scientific detector audit usedtop5IoU;thisVroutebudgetusesatmost3boxes,soperformancecannotbeassumedidentical","test_calibration_or_modelvalidation_used":False,"elapsed_seconds":time.perf_counter()-started,"scientific_proposal_or_VLP_training_released":False}
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("provenance","rows")},indent=2))


if __name__=="__main__":
    main()
