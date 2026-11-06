"""Fixed proposal budget audit and explicitly incomplete TRAIN fixture artifact."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scripts.train_detector_v2 import BoPhatHienDataset,iou_hop
from src.preprocessing import canonicalize_raster,RASTER_PREPROCESSING_VERSION
from src.provenance import validate_detector_proposals
from src.detector_protocol import load_detector_manifest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scoped_check(path):
    artifact=validate_detector_proposals(path,require_complete=False)
    side=json.loads(path.with_suffix(".provenance.json").read_text())
    assert artifact["scope"]=="TRAIN_technical_fixture_only" and artifact["complete"] is False
    manifest,ids,_=load_detector_manifest(artifact["split_manifest"])
    assert set(side["images"])<=set(ids["train"])
    assert artifact["detector_transform"]=={"min_size":[640],"max_size":640,"size_divisible":32}
    for key in ("scoped_runtime","scoped_loader","TRAIN_runtime_gate"):
        assert sha(artifact[key])==artifact[key+"_sha256"]
    assert artifact["max_boxes"]==3 and artifact["score_threshold"]==0.05
    for value in side["images"].values():
        assert value["split"]=="train" and sha(value["input_image_path"])==value["input_image_sha256"]
    return artifact


def main():
    started=time.perf_counter()
    root=Path.cwd()
    parent=root/"refactor_v2/reports/detector640_fold0_bounded_quality_v3_1"
    folder=root/"refactor_v2/reports/detector640_top3_binding_v3"
    assert not folder.exists()
    evaluation=parent/"exact_architecture_independent_verification.json"
    assert sha(evaluation)==json.loads((parent/"exact_architecture_verification_progress.json").read_text())["report_sha256"]
    report=json.loads(evaluation.read_text())
    release=json.loads((parent/"execution_release.json").read_text())
    manifest,ids,digest=load_detector_manifest(release["manifest_path"])
    ds=BoPhatHienDataset("val",canh_dai=640,allowed_ids=set(ids["val"]))
    paths={iid:Path(ds.root)/item[0] for iid,item in ds.muc}
    rows=[]
    for row in report["results"]["640"]["records"]:
        with Image.open(paths[row["image_id"]]) as image:
            w,h=canonicalize_raster(image).size
        scale=640/max(w,h)
        px=(640-max(1,int(round(w*scale))))//2
        py=(640-max(1,int(round(h*scale))))//2
        raw=[box for box,score in row["automatic_boxes_scores"]]
        clipped=[]
        for box in raw:
            x1=max(0,min(w,(box[0]-px)/scale));y1=max(0,min(h,(box[1]-py)/scale))
            x2=max(0,min(w,(box[2]-px)/scale));y2=max(0,min(h,(box[3]-py)/scale))
            if x2>x1 and y2>y1:
                clipped.append([x1*scale+px,y1*scale+py,x2*scale+px,y2*scale+py])
        values={}
        for label,boxes in (("raw_top3",raw[:3]),("raw_top5",raw[:5]),("runtime_clipped_top3",clipped[:3]),("runtime_clipped_top5",clipped[:5])):
            values[label]={"nonempty":bool(boxes),"bestIoU":max((iou_hop(b,g) for b in boxes for g in row["GT_boxes"]),default=0.0) if row["positive"] else None}
        rows.append({"image_id":row["image_id"],"positive":row["positive"],"conditions":values})
    metrics={}
    for condition in rows[0]["conditions"]:
        pos=[r["conditions"][condition] for r in rows if r["positive"]]
        neg=[r["conditions"][condition] for r in rows if not r["positive"]]
        metrics[condition]={"positive_images":len(pos),"negative_images":len(neg),"mean_positive_bestIoU":float(np.mean([r["bestIoU"] for r in pos])),"positive_images_IoU_ge0_5":sum(r["bestIoU"]>=0.5 for r in pos),"positive_zero_overlap":sum(r["bestIoU"]==0 for r in pos),"negative_any_proposal":sum(r["nonempty"] for r in neg)}
    assert round(metrics["raw_top5"]["mean_positive_bestIoU"],4)==0.416 and metrics["raw_top5"]["positive_images_IoU_ge0_5"]==28
    # Export only already-verified eight TRAIN fixtures,never modelval/test.
    fixture_path=parent/"TRAIN_image_runtime_gate.json"
    fixtures=json.loads(fixture_path.read_text())
    assert fixtures["passed"]
    train=BoPhatHienDataset("train",canh_dai=640,allowed_ids=set(ids["train"]))
    train_paths={iid:Path(train.root)/item[0] for iid,item in train.muc}
    folder.mkdir()
    proposals={r["TRAIN_image_id"]:r["boxes"] for r in fixtures["rows"]}
    proposal_path=folder/"TRAIN_technical_boxes.json"
    proposal_path.write_text(json.dumps(proposals,indent=2)+"\n")
    artifact={"schema_version":3,"scope":"TRAIN_technical_fixture_only","complete":False,"coordinate_space":"exif_canonical_pixels","preprocessing_version":RASTER_PREPROCESSING_VERSION,"checkpoint":fixtures["provenance"]["checkpoint"],"checkpoint_sha256":fixtures["provenance"]["checkpoint_sha256"],"split_manifest":release["manifest_path"],"split_manifest_sha256":digest,"proposal_sha256":sha(proposal_path),"coverage":len(proposals),"expected_coverage":len(set().union(*(set(v) for v in ids.values()))),"score_threshold":0.05,"max_boxes":3,"detector_transform":release["effective_transform"],"normalization_spec_sha256":fixtures["provenance"]["normalization_spec_sha256"],"scoped_runtime":str(root/"refactor_v2/experiments/detector640_scoped_runtime_v3.py"),"scoped_loader":str(root/"refactor_v2/experiments/detector640_checkpoint_v3.py"),"TRAIN_runtime_gate":str(fixture_path)}
    for key in ("scoped_runtime","scoped_loader","TRAIN_runtime_gate"):
        artifact[key+"_sha256"]=sha(artifact[key])
    images={r["TRAIN_image_id"]:{"split":"train","width":r["canonical_image_size"][0],"height":r["canonical_image_size"][1],"input_image_path":str(train_paths[r["TRAIN_image_id"]]),"input_image_sha256":sha(train_paths[r["TRAIN_image_id"]])} for r in fixtures["rows"]}
    side={"artifact":artifact,"images":images}
    proposal_path.with_suffix(".provenance.json").write_text(json.dumps(side,indent=2)+"\n")
    scoped_check(proposal_path)
    try:
        validate_detector_proposals(proposal_path,require_complete=True)
    except RuntimeError as error:
        assert "chưa phủ toàn bộ sealed splits" in str(error)
        production_rejection=str(error)
    else:
        raise RuntimeError("Incomplete technical proposals unexpectedly passed production guard")
    # Negative fixture retains valid JSON/proposal hash but wrong transform.
    bad=folder/"negative_fixture_wrong_transform.json"
    bad.write_text(proposal_path.read_text())
    wrong=json.loads(json.dumps(side))
    wrong["artifact"]["detector_transform"]["min_size"]=[320]
    bad.with_suffix(".provenance.json").write_text(json.dumps(wrong,indent=2)+"\n")
    try:
        scoped_check(bad)
    except AssertionError:
        wrong_transform_rejected=True
    else:
        raise RuntimeError("Wrong transform passed scoped check")
    result={"passed":True,"scope":"cachedmodelvalbudgetaudit_and8TRAINtechnicalfixturebinding;noqualitytrainingrelease","cached_evaluation_sha256":sha(evaluation),"fixed_threshold":0.05,"top3vs5_metrics":metrics,"rows":rows,"technical_proposal_sha256":sha(proposal_path),"technical_sidecar_sha256":sha(proposal_path.with_suffix(".provenance.json")),"production_complete_guard_rejection":production_rejection,"negative_wrong_transform_rejected":wrong_transform_rejected,"elapsed_seconds":time.perf_counter()-started,"test_or_calibration_used":False,"scientific_VLP_or_production_released":False,"limitations":["GTonlyevaluatescachedpredictions,notinputtoproposalgeneration","8TRAINfixturesincompleteandcannotbesuppliedtoscientificCLI","budgetauditnotclinicalvalidation;nohyperparametersearch"]}
    (folder/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":
    main()
