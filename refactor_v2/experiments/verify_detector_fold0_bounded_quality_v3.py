"""Independent strict detector reload/modelvalidation box audit,no test fitting."""
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.train_detector_v2 import BoPhatHienDataset,collate,tao_mo_hinh,iou_hop
from src.detector_protocol import load_detector_manifest,verify_detector_checkpoint_binding,verify_detector_annotation_parity
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def main():
    started=time.perf_counter()
    root=Path.cwd()
    folder=root/"refactor_v2/reports/detector_fold0_bounded_quality_v3"
    checkpoints=root/"refactor_v2/detector_fold0_bounded_quality_v3"
    output=folder/"independent_verification.json"
    progress=folder/"verification_progress.json"
    assert not output.exists() and not progress.exists()
    state={"status":"verifying","scope":"modelvalidation detector diagnostic only,no proposal or VLP release"}
    def save():
        progress.write_text(json.dumps(state,indent=2)+"\n")
    save()
    try:
        done=json.loads((folder/"completion.json").read_text())
        prior=json.loads((folder/"progress.json").read_text())
        assert done["complete"] and sha(folder/"completion.json")==prior["completion_sha256"]
        release=json.loads((folder/"execution_release.json").read_text())
        payload,ids,digest=load_detector_manifest(release["manifest_path"])
        assert digest==done["manifest_sha256"]==release["manifest_sha256"]
        artifacts={}
        for label in ("last","best"):
            path=checkpoints/(label+".pt")
            assert sha(path)==done[label+"_sha256"]
            artifacts[label]=torch.load(path,map_location="cpu",weights_only=False)
            verify_detector_checkpoint_binding(artifacts[label],payload,digest)
            assert artifacts[label]["training_image_ids"]==sorted(ids["train"])
            assert finite_tree(artifacts[label]["state"])
            assert sha(artifacts[label]["initial_weight_provenance"]["path"])==artifacts[label]["initial_weight_provenance"]["sha256"]
        assert artifacts["last"]["epoch"]==2 and artifacts["last"]["optimizer_steps"]==1408
        optimizer=torch.load(checkpoints/"optimizer.pt",map_location="cpu",weights_only=False)
        assert sha(checkpoints/"optimizer.pt")==done["optimizer_sha256"] and finite_tree(optimizer) and optimizer["optimizer_steps"]==1408
        dataset=BoPhatHienDataset("val",canh_dai=320,allowed_ids=set(ids["val"]))
        assert len(dataset)==321
        verify_detector_annotation_parity(dataset,payload["source_vqa"]["val"]["path"])
        loader=DataLoader(dataset,batch_size=4,num_workers=0,shuffle=False,collate_fn=collate)
        torch.set_num_threads(8)
        model=tao_mo_hinh("mobilenet_320",weights=None).eval()
        model.load_state_dict(artifacts["last"]["state"],strict=True)
        rng=random.Random(20261020)
        rows=[]
        for images,targets,image_ids in loader:
            predictions=model(list(images))
            for iid,pred,target in zip(image_ids,predictions,targets):
                ground=target["boxes"].tolist()
                candidates=[(b,float(s)) for b,s in zip(pred["boxes"].tolist(),pred["scores"].tolist()) if s>=0.05]
                boxes=[b for b,_ in candidates]
                actual=max((iou_hop(p,g) for p in boxes[:5] for g in ground),default=0.0)
                # Fixed single geometry placebo: same number/width/height,random
                # positions on320canvas;not tuned and not a semantic-null proof.
                placebo=[]
                for x1,y1,x2,y2 in boxes[:5]:
                    w,h=min(x2-x1,320.0),min(y2-y1,320.0)
                    x,y=rng.uniform(0,max(0,320-w)),rng.uniform(0,max(0,320-h))
                    placebo.append([x,y,x+w,y+h])
                null=max((iou_hop(p,g) for p in placebo for g in ground),default=0.0)
                rows.append({"image_id":iid,"GT_boxes":ground,"automatic_boxes_scores":candidates,"positive_best_top5_IoU":actual if ground else None,"positive_random_position_best_top5_IoU":null if ground else None})
            state.update(status="running",completed_images=len(rows))
            save()
        positives=[r for r in rows if r["GT_boxes"]]
        negatives=[r for r in rows if not r["GT_boxes"]]
        hits=sum(r["positive_best_top5_IoU"]>=0.5 for r in positives)
        missed=sum(r["positive_best_top5_IoU"]==0 for r in positives)
        fp=sum(bool(r["automatic_boxes_scores"]) for r in negatives)
        metrics={"so_anh":321,"so_anh_duong":len(positives),"so_anh_am":len(negatives),"iou_tot_nhat_tb":round(float(np.mean([r["positive_best_top5_IoU"] for r in positives])),4),"ty_le_iou_tren_0_5":round(hits/len(positives),4),"ty_le_khong_tim_thay":round(missed/len(positives),4),"negative_specificity":round(1-fp/len(negatives),4),"negative_false_positive_rate":round(fp/len(negatives),4)}
        assert all(metrics[k]==done["history"][-1][k] for k in metrics)
        positive_records=[r for r in dataset.muc if r[1][1]]
        report={"passed_artifact_and_metric_reproduction":True,"LAST_checkpoint_sha256":done["last_sha256"],"BEST_epoch":artifacts["best"]["epoch"],"BEST_LAST_weights_identical":all(torch.equal(v,artifacts["best"]["state"][k]) for k,v in artifacts["last"]["state"].items()),"training_ids_and_full_fold_binding_exact":True,"weights_optimizer_finite":True,"reproduced_metrics":metrics,"positive_IoU_ge0_5_count":hits,"positive_zero_overlap_count":missed,"negative_any_proposal_count":fp,"automatic_proposal_image_coverage":sum(bool(r["automatic_boxes_scores"]) for r in rows)/321,"geometry_placebo_fixed_seed":{"positive_mean_best_top5_IoU":float(np.mean([r["positive_random_position_best_top5_IoU"] for r in positives])),"positive_IoU_ge0_5_count":sum(r["positive_random_position_best_top5_IoU"]>=0.5 for r in positives)},"records":rows,"elapsed_seconds":time.perf_counter()-started,"scope":state["scope"],"score_threshold":0.05,"test_or_calibration_evaluated":False,"VLP_or_production_proposal_released":False,"interpretation":"Execution/provenance reproduce,but localization coverage weak;nonempty boxes are not correct detections. No score threshold/epoch search or clinical claims.","limitations":["best-IoU-per-positive-image metric is not lesion recall or mAP","single fixed random-position placebo is diagnostic only;letterbox padding included","twoepochs one seed on inspecteddevelopment,not detector efficacy confirmation"]}
        output.write_text(json.dumps(report,indent=2)+"\n")
        state.update(status="complete_pending_quality_interpretation",report_sha256=sha(output))
        save()
        print(json.dumps({k:v for k,v in report.items() if k!="records"},indent=2))
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry",failure=repr(error))
        save()
        raise


if __name__=="__main__":
    main()
