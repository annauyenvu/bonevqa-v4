"""Fresh effective640 two-epoch comparison;fail-closed transform reload gate."""
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.train_detector_v2 import BoPhatHienDataset,PositiveAnchoredBatchSampler,collate,tao_mo_hinh,danh_gia
from src.detector_protocol import load_detector_manifest,verify_detector_annotation_parity
from src.preprocessing import RASTER_PREPROCESSING_VERSION
from src.utils import set_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    root=Path.cwd()
    folder=root/"refactor_v2/reports/detector640_fold0_bounded_quality_v3_1"
    checkpoints=root/"refactor_v2/detector640_fold0_bounded_quality_v3_1"
    gate_folder=root/"refactor_v2/reports/detector_effective_resolution_gate_v3"
    baseline_folder=root/"refactor_v2/reports/detector_fold0_bounded_quality_v3"
    if folder.exists() or checkpoints.exists():
        raise RuntimeError("Existing640pilot:preserve and inspect,never overwrite")
    commands=subprocess.check_output(["ps","-axo","command"],text=True)
    if any(" -m src.train " in row or "run_detector_fold0" in row or "scripts/train_detector_v2.py" in row for row in commands.splitlines()):
        raise RuntimeError("Existing training active")
    assert os.environ["HF_HUB_OFFLINE"]==os.environ["TRANSFORMERS_OFFLINE"]=="1" and os.environ["OMP_NUM_THREADS"]=="8"
    subprocess.run([sys.executable,"refactor_v2/scripts/seal_technical_reference_v3.py","--verify"],check=True)
    gate=json.loads((gate_folder/"report.json").read_text())
    protocol=json.loads((gate_folder/"next_bounded_protocol.json").read_text())
    assert gate["passed"] and sha(gate_folder/"report.json")==protocol["technical_timing_report_sha256"]
    baseline=json.loads((baseline_folder/"execution_release.json").read_text())
    manifest=Path(baseline["manifest_path"])
    payload,ids,digest=load_detector_manifest(manifest)
    assert digest==gate["fold_manifest_sha256"]==baseline["manifest_sha256"]
    train=BoPhatHienDataset("train",canh_dai=640,tang_cuong=True,allowed_ids=set(ids["train"]))
    val=BoPhatHienDataset("val",canh_dai=640,allowed_ids=set(ids["val"]))
    assert len(train)==2568 and len(val)==321
    for role,ds in (("train",train),("val",val)):
        verify_detector_annotation_parity(ds,payload["source_vqa"][role]["path"])
    set_seed(42)
    torch.set_num_threads(8)
    sampler=PositiveAnchoredBatchSampler(train,4,seed=42)
    loader=DataLoader(train,batch_sampler=sampler,collate_fn=collate,num_workers=0)
    validation=DataLoader(val,batch_size=4,shuffle=False,collate_fn=collate,num_workers=0)
    schedule=[]
    for epoch in range(2):
        sampler.epoch=epoch
        batches=list(iter(sampler))
        assert sorted(set(i for b in batches for i in b))==list(range(2568))
        digest_order=hashlib.sha256(json.dumps(batches).encode()).hexdigest()
        assert digest_order==baseline["sampler"][epoch]["batch_order_sha256"] and len(batches)==704
        schedule.append(digest_order)
    sampler.epoch=0
    folder.mkdir()
    checkpoints.mkdir()
    state={"status":"preparing_transform_reload_gate","pid":os.getpid(),"started_utc":now(),"scope":"fresh640one-factor2epoch1408stepdetectorpilot;not VLP or production release"}
    def save():
        (folder/"progress.json").write_text(json.dumps(state,indent=2)+"\n")
    save()
    try:
        model=tao_mo_hinh("mobilenet_320").to("cpu")
        hashes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
        assert hashes==gate["same_initial_state_tensor_hashes"]
        provenance=model.initial_weight_provenance
        assert provenance["sha256"]==baseline["initial_weight_provenance"]["sha256"]==sha(provenance["path"])
        model.transform.min_size=(640,)
        model.transform.max_size=640
        normalization={k:{"type":type(v).__name__,"eps":v.eps} for k,v in model.named_modules() if hasattr(v,"running_mean") and hasattr(v,"eps")}
        assert normalization and all(v["type"]=="FrozenBatchNorm2d" for v in normalization.values())
        transform={"min_size":[640],"max_size":640,"size_divisible":model.transform.size_divisible}
        assert transform["size_divisible"]==32
        def checkpoint(scores,epoch,step):
            return {"state":model.state_dict(),"canh_dai":640,"detector_transform":transform,"normalization_spec":normalization,"val":scores,"architecture":"mobilenet_320","includes_negative_images":True,"training_split_sha256":digest,"training_scope":payload["scope"],"training_image_ids":sorted(ids["train"]),"initial_weight_provenance":provenance,"preprocessing_version":RASTER_PREPROCESSING_VERSION,"pilot_scope":state["scope"],"epoch":epoch,"optimizer_steps":step,"production_export_released":False}
        # Same four TRAIN images in parent/fresh interpreter;no targets to infer.
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        fixture_ids=gate["results"]["640"]["TRAIN_image_order"][0]
        fixture_ds=BoPhatHienDataset("train",canh_dai=640,tang_cuong=False,allowed_ids=set(fixture_ids))
        fixture_images=[fixture_ds[i][0] for i in range(len(fixture_ds))]
        (folder/"reload_fixture.json").write_text(json.dumps({"TRAIN_image_ids":sorted(fixture_ids)},indent=2)+"\n")
        initial_path=checkpoints/"PREtask_initialization.pt"
        torch.save(checkpoint({},0,0),initial_path)
        model.eval()
        with torch.no_grad():
            expected=model(fixture_images)
        subprocess.run([sys.executable,"refactor_v2/experiments/detector640_checkpoint_v3.py",str(initial_path),str(folder/"reload_fixture.json"),str(folder/"fresh_process_reload.pt")],check=True)
        reloaded=torch.load(folder/"fresh_process_reload.pt",map_location="cpu",weights_only=False)
        errors={k:max(float((a[k]-b[k]).abs().max()) if a[k].numel() else 0.0 for a,b in zip(expected,reloaded["predictions"])) for k in ("boxes","scores","labels")}
        assert all(v==0 for v in errors.values()) and all(tuple(s)==(640,640) for s in reloaded["actual_image_sizes"])
        assert {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}==hashes
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.SGD(parameters,lr=0.005,momentum=0.9,weight_decay=0.0005)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2)
        release={"passed":True,"scope":state["scope"],"protocol_sha256":sha(gate_folder/"next_bounded_protocol.json"),"manifest_path":str(manifest),"manifest_sha256":digest,"TRAIN_images":2568,"modelvalidation_images":321,"source_script_sha256":sha(__file__),"scoped_loader_sha256":sha(root/"refactor_v2/experiments/detector640_checkpoint_v3.py"),"all284initial_tensors_exact":hashes,"PREtask_initialization_sha256":sha(initial_path),"initial_weight_provenance":provenance,"effective_transform":transform,"normalization_spec":normalization,"fresh_process_reload_errors":errors,"actual_reload_tensor_shape":reloaded["tensor_shape"],"same320_epoch_batch_order_sha256":schedule,"seed":42,"epochs_cap":2,"optimizer_steps_cap":1408,"optimizer":"SGD0.005,momentum0.9,wd0.0005,clip5,cosine2epochs","device":"CPU8float32workers0","evaluation_threshold":0.05,"primary":"LASTepoch2","test_or_calibration_evaluated":False,"compute_matching":"sameupdatesNOTsameFLOPs/time;recordactualtiming","VLP_or_production_released":False,"long_training_released":False}
        (folder/"execution_release.json").write_text(json.dumps(release,indent=2)+"\n")
        del expected,reloaded,fixture_ds,fixture_images
        started=time.perf_counter()
        history=[]
        step=0
        best=-1.0
        selected=model.roi_heads.box_predictor.cls_score.weight
        before=selected.detach().clone()
        for epoch in range(2):
            t0=time.perf_counter()
            model.train()
            losses=[]
            for images,targets,_ in loader:
                terms=model(list(images),list(targets))
                loss=sum(terms.values())
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if step==0:
                    grad=float(selected.grad.abs().max())
                    assert grad>0
                    state["first_step_classifier_gradient_max"]=grad
                torch.nn.utils.clip_grad_norm_(parameters,5.0)
                optimizer.step()
                step+=1
                losses.append(float(loss.detach()))
                if step==1:
                    update=float((selected.detach()-before).abs().max())
                    assert update>0
                    state["first_step_classifier_update_max"]=update
                if step==1 or step%25==0:
                    state.update(status="training",epoch=epoch+1,optimizer_step=step,maximum_optimizer_steps=1408,latest_total_loss=losses[-1],elapsed_seconds=time.perf_counter()-started)
                    save()
            scheduler.step()
            scores=danh_gia(model,validation,torch.device("cpu"),nguong=0.05)
            scores.update(epoch=epoch+1,mean_training_loss=sum(losses)/len(losses),epoch_seconds=time.perf_counter()-t0,optimizer_steps=step)
            history.append(scores)
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            torch.save(checkpoint(scores,epoch+1,step),checkpoints/"last.pt")
            torch.save({"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"optimizer_steps":step},checkpoints/"optimizer.pt")
            if scores["iou_tot_nhat_tb"]>best:
                best=scores["iou_tot_nhat_tb"]
                torch.save(checkpoint(scores,epoch+1,step),checkpoints/"best.pt")
            state.update(status="epoch_complete",completed_epochs=epoch+1,history=history)
            save()
        done={"complete":True,"scope":state["scope"],"history":history,"wall_seconds":time.perf_counter()-started,"optimizer_steps":step,"last_sha256":sha(checkpoints/"last.pt"),"best_sha256":sha(checkpoints/"best.pt"),"optimizer_sha256":sha(checkpoints/"optimizer.pt"),"manifest_sha256":digest,"effective_transform":transform,"test_or_calibration_evaluated":False,"quality_gate_passed":False,"pending":"Independent trainedcheckpoint transform/provenance/metric verification beforeVLPdecision"}
        (folder/"completion.json").write_text(json.dumps(done,indent=2)+"\n")
        state.update(status="two_epochs_complete_pending_independent_verification",finished_utc=now(),completion_sha256=sha(folder/"completion.json"))
        save()
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry",failure=repr(error),finished_utc=now())
        save()
        raise


if __name__=="__main__":
    main()
