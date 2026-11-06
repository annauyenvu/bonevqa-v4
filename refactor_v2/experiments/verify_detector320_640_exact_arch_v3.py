"""Exact training-time normalization/transform reload for320/640LAST pilots."""
import gc
import hashlib
import json
import os
import random
import subprocess
import time
from collections import defaultdict
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
    root=Path.cwd()
    folders={320:root/"refactor_v2/reports/detector_fold0_bounded_quality_v3",640:root/"refactor_v2/reports/detector640_fold0_bounded_quality_v3_1"}
    for folder in folders.values():
        assert json.loads((folder/"progress.json").read_text())["status"]=="two_epochs_complete_pending_independent_verification"
    commands=subprocess.check_output(["ps","-axo","command"],text=True)
    if any("run_detector640_fold0_bounded_quality_v3_1.py" in row or " -m src.train " in row for row in commands.splitlines()):
        raise RuntimeError("Training active,do not compete")
    output=folders[640]/"exact_architecture_independent_verification.json"
    progress=folders[640]/"exact_architecture_verification_progress.json"
    assert not output.exists() and not progress.exists()
    state={"status":"verifying","pid":os.getpid(),"scope":"Same321modelvalidationimages,no calibration/test;not independent clinical confirmation"}
    def save():
        progress.write_text(json.dumps(state,indent=2)+"\n")
    save()
    started=time.perf_counter()
    try:
        torch.set_num_threads(8)
        results={}
        full_ids=None
        components=None
        for size,folder in folders.items():
            done=json.loads((folder/"completion.json").read_text())
            assert sha(folder/"completion.json")==json.loads((folder/"progress.json").read_text())["completion_sha256"]
            release=json.loads((folder/"execution_release.json").read_text())
            payload,ids,digest=load_detector_manifest(release["manifest_path"])
            assert digest==done["manifest_sha256"]==release["manifest_sha256"]
            components=payload["component_by_image"]
            path=root/("refactor_v2/detector_fold0_bounded_quality_v3" if size==320 else "refactor_v2/detector640_fold0_bounded_quality_v3_1")
            checkpoints={}
            for label in ("last","best"):
                assert sha(path/(label+".pt"))==done[label+"_sha256"]
                checkpoints[label]=torch.load(path/(label+".pt"),map_location="cpu",weights_only=False)
                verify_detector_checkpoint_binding(checkpoints[label],payload,digest)
                assert checkpoints[label]["training_image_ids"]==sorted(ids["train"]) and finite_tree(checkpoints[label]["state"])
            last=checkpoints["last"]
            assert last["epoch"]==2 and last["optimizer_steps"]==1408 and last["canh_dai"]==size
            assert sha(path/"optimizer.pt")==done["optimizer_sha256"]
            optimizer=torch.load(path/"optimizer.pt",map_location="cpu",weights_only=False)
            assert finite_tree(optimizer) and optimizer["optimizer_steps"]==1408
            source=last["initial_weight_provenance"]
            assert sha(source["path"])==source["sha256"]
            # Use exact training-time pretrained constructor (local cachedCOCO),
            # NOT scratch weightsNone BatchNorm architecture from earlier audit.
            model=tao_mo_hinh("mobilenet_320").to("cpu").eval()
            norms={k:{"type":type(v).__name__,"eps":v.eps} for k,v in model.named_modules() if hasattr(v,"running_mean") and hasattr(v,"eps")}
            assert norms and all(v["type"]=="FrozenBatchNorm2d" for v in norms.values())
            if size==640:
                assert norms==last["normalization_spec"]==release["normalization_spec"]
                transform=last["detector_transform"]
                assert transform==release["effective_transform"]=={"min_size":[640],"max_size":640,"size_divisible":32}
                model.transform.min_size=tuple(transform["min_size"])
                model.transform.max_size=transform["max_size"]
                model.transform.size_divisible=transform["size_divisible"]
            else:
                assert model.transform.min_size==(320,) and model.transform.max_size==640
            model.load_state_dict(last["state"],strict=True)
            ds=BoPhatHienDataset("val",canh_dai=size,allowed_ids=set(ids["val"]))
            assert len(ds)==321
            verify_detector_annotation_parity(ds,payload["source_vqa"]["val"]["path"])
            loader=DataLoader(ds,batch_size=4,num_workers=0,shuffle=False,collate_fn=collate)
            rows=[]
            actual_geometry=None
            for images,targets,iids in loader:
                if actual_geometry is None:
                    transformed,_=model.transform(list(images),None)
                    assert all(tuple(s)==(size,size) for s in transformed.image_sizes)
                    actual_geometry={"tensor_shape":list(transformed.tensors.shape),"image_sizes":transformed.image_sizes}
                predictions=model(list(images))
                for iid,pred,target in zip(iids,predictions,targets):
                    ground=target["boxes"].tolist()
                    detections=[(b,float(s)) for b,s in zip(pred["boxes"].tolist(),pred["scores"].tolist()) if s>=0.05]
                    overlap=max((iou_hop(b,g) for b,_ in detections[:5] for g in ground),default=0.0)
                    rows.append({"image_id":iid,"positive":bool(ground),"GT_boxes":ground,"automatic_boxes_scores":detections,"positive_best_top5_IoU":overlap if ground else None})
                state.update(status="running",resolution=size,completed_images=len(rows))
                save()
            positive=[r for r in rows if r["positive"]]
            negative=[r for r in rows if not r["positive"]]
            hit=sum(r["positive_best_top5_IoU"]>=0.5 for r in positive)
            zero=sum(r["positive_best_top5_IoU"]==0 for r in positive)
            fp=sum(bool(r["automatic_boxes_scores"]) for r in negative)
            metrics={"so_anh":321,"so_anh_duong":len(positive),"so_anh_am":len(negative),"iou_tot_nhat_tb":round(float(np.mean([r["positive_best_top5_IoU"] for r in positive])),4),"ty_le_iou_tren_0_5":round(hit/len(positive),4),"ty_le_khong_tim_thay":round(zero/len(positive),4),"negative_specificity":round(1-fp/len(negative),4),"negative_false_positive_rate":round(fp/len(negative),4)}
            assert all(v==done["history"][-1][k] for k,v in metrics.items())
            cohort=[(r["image_id"],r["positive"]) for r in rows]
            if full_ids is None:
                full_ids=cohort
            else:
                assert cohort==full_ids
            results[str(size)]={"metrics_exactly_reproduced":metrics,"positive_hit_count":hit,"positive_zero_overlap_count":zero,"negative_any_proposal_count":fp,"normalization_types_eps":norms,"actual_geometry":actual_geometry,"training_wall_seconds":done["wall_seconds"],"LAST_sha256":done["last_sha256"],"BEST_epoch":checkpoints["best"]["epoch"],"records":rows}
            del model,last,checkpoints,optimizer
            gc.collect()
        groups=defaultdict(list)
        for i,(iid,_) in enumerate(full_ids):
            groups[components[iid]].append(i)
        keys=sorted(groups)
        rng=random.Random(20261020)
        differences={k:[] for k in ("positive_mean_bestIoU","positive_hit_fraction","negative_false_proposal_fraction")}
        for _ in range(500):
            indices=[i for key in rng.choices(keys,k=len(keys)) for i in groups[key]]
            scores={}
            for size in ("320","640"):
                selected=[results[size]["records"][i] for i in indices]
                pos=[r for r in selected if r["positive"]];neg=[r for r in selected if not r["positive"]]
                scores[size]={"positive_mean_bestIoU":float(np.mean([r["positive_best_top5_IoU"] for r in pos])),"positive_hit_fraction":sum(r["positive_best_top5_IoU"]>=0.5 for r in pos)/len(pos),"negative_false_proposal_fraction":sum(bool(r["automatic_boxes_scores"]) for r in neg)/len(neg)}
            for k in differences:
                differences[k].append(scores["640"][k]-scores["320"][k])
        report={"passed_artifact_and_exact_architecture_metric_checks":True,"scope":state["scope"],"results":results,"paired_component_bootstrap95_640minus320_descriptive":{k:np.percentile(v,[2.5,97.5]).tolist() for k,v in differences.items()},"elapsed_seconds":time.perf_counter()-started,"fixed_score_threshold":0.05,"test_or_calibration_evaluated":False,"VLP_or_production_released":False,"limitations":["bestIoUperpositiveimage is not lesionrecall/mAP/clinical sensitivity","single seed2epochs on inspecteddevelopment;bootstrap descriptive,not independent confirmation","1408updatesmatched,640highercompute/time;no threshold/epoch search","previous320weightsNoneBatchNormaudit preserved;thiscorrectFrozenBatchNormaudit supersedes architecture-equivalence claim"]}
        output.write_text(json.dumps(report,indent=2)+"\n")
        state.update(status="complete_pending_quality_interpretation",report_sha256=sha(output))
        save()
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry",failure=repr(error))
        save()
        raise


if __name__=="__main__":
    main()
