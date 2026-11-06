"""Fixed-LAST P/control evaluation,model-validation only;no threshold fitting."""
import gc
import hashlib
import json
import os
import random
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from src.context_controls import load_component_lookup
from src.data import build_dataset, collate_fn
from src.models.bonevqa import BoneVQAModel
from src.protocol import manifest, records_of
from src.train import build_split
from src.utils import compute_metrics, normalize_answer, set_seed
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def main():
    started=time.perf_counter()
    root=Path.cwd()
    folder=root/"refactor_v2/reports/P_target_semantic_pilot_v3"
    parent=root/"refactor_v2/reports/fracatlas_scale_pilot_v3"
    assert json.loads((folder/"progress.json").read_text())["status"]=="three_arms_complete_pending_independent_evaluation"
    if any(" -m src.train " in row for row in subprocess.check_output(["ps","-axo","command"],text=True).splitlines()):
        raise RuntimeError("Training active;do not compete")
    progress=folder/"independent_evaluation_progress.json"
    output=folder/"independent_evaluation.json"
    if progress.exists() or output.exists():
        raise RuntimeError("Existing evaluation;preserve,never overwrite")
    state={"status":"verifying","pid":os.getpid(),"started_utc":datetime.now(timezone.utc).isoformat(),"scope":"Single-seed inspected development P semantic screening,not independent clinical evidence"}
    def save():
        progress.write_text(json.dumps(state,indent=2)+"\n")
    save()
    try:
        release=json.loads((folder/"execution_release.json").read_text())
        assert release["passed"] and sha(folder/"protocol.json")==release["protocol_sha256"]
        old_state=json.loads((parent/"independent_evaluation_progress.json").read_text())
        assert sha(parent/"independent_evaluation.json")==old_state["report_sha256"]
        old=json.loads((parent/"independent_evaluation.json").read_text())
        bname="fracatlas_B_roco_scale_pilot_v3"
        bfull=old["full321_fracture"][bname]
        names=list(release["config_sha256"])
        assert len(names)==3
        cfg=json.loads((folder/(names[0]+".json")).read_text())
        datasets={role:build_split(build_dataset,cfg["data"]["dataset"],role,None,True,cfg["data"]["loader_kwargs"]) for role in ("train","model_val")}
        assert {k:manifest(v) for k,v in datasets.items()}==release["actual_manifests"]
        train,pilot=datasets["train"],datasets["model_val"]
        base=json.loads((root/"refactor_v2/configs/revised_sealed_candidates_v3/b.json").read_text())
        full=build_split(build_dataset,base["data"]["dataset"],"model_val",None,True,base["data"]["loader_kwargs"])
        canonical="Is there a fracture in this X-ray?"
        indices=[i for i,r in enumerate(records_of(full)) if r["question"]==canonical]
        assert len(indices)==321 and len(pilot)==142
        bc=cfg["train"]["component_batching"]
        lookup,_=load_component_lookup(bc["manifest"],bc["seal"],root)
        first={}
        for i,row in enumerate(records_of(pilot)):
            first.setdefault(row["image_id"],i)
        ids=sorted(first)
        donors=dict(zip(ids,ids[1:]+ids[:1]))
        assert all(lookup[i]!=lookup[d] for i,d in donors.items())
        pixels={iid:pilot[i]["image"] for iid,i in first.items()}
        questions=json.loads((root/"refactor_v2/reports/fracatlas_baseline_pilot_v3/independent_paired_evaluation.json").read_text())["nonconstant_questions_defined_from_TRAIN"]
        def metrics(rows):
            result=compute_metrics(rows,fracture_threshold=0.5)
            byq=defaultdict(list)
            for row in rows:
                byq[row["question"]].append(row)
            result["per_question"]={q:{"n":len(group),"exact":sum(normalize_answer(r["pred"])==normalize_answer(r["gt"]) for r in group)/len(group)} for q,group in byq.items()}
            result["macro_question_exact"]=sum(v["exact"] for v in result["per_question"].values())/len(byq)
            result["shared_original_TRAIN_nonconstant_metrics"]=compute_metrics([r for r in rows if r["question"] in questions],fracture_threshold=0.5)
            return result
        results,checks={},{}
        for name in names:
            run=root/"refactor_v2/checkpoints"/name
            done=json.loads((folder/(name+"_completion.json")).read_text())
            assert sha(folder/(name+".json"))==release["config_sha256"][name]
            assert sha(run/"last.pt")==done["last_sha256"] and sha(run/"best.pt")==done["best_sha256"]
            init_path=folder/(name+"_initialization.pt")
            assert sha(init_path)==release["initialization_sha256"][name]
            initial=torch.load(init_path,map_location="cpu",weights_only=False)["state"]
            assert {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in initial.items()}==release["initial_tensor_hashes"][name]
            last=torch.load(run/"last.pt",map_location="cpu",weights_only=False)
            best=torch.load(run/"best.pt",map_location="cpu",weights_only=False)
            optimizer=torch.load(run/"optimizer.pt",map_location="cpu",weights_only=False)
            assert finite_tree(last["state"]) and finite_tree(best["state"]) and finite_tree(optimizer)
            assert last["extra"]["global_step"]==optimizer["global_step"]==296 and last["extra"]["epoch"]==4
            stored=json.loads((run/"data_manifest.json").read_text())
            assert stored["train"]==release["actual_manifests"]["train"] and stored["val"]==release["actual_manifests"]["model_val"]
            selected=("fusion.gamma_latent","latent.latent","closed_head.net.3.weight","fracture_head.net.3.weight")
            updates={k:float((last["state"][k]-initial[k]).abs().max()) for k in selected}
            assert all(v>0 for v in updates.values())
            checks[name]={"passed":True,"selected_actual_updates":updates,"finite_weights_optimizer":True,"last296_steps":True,"actual_training_seconds":done["wall_seconds"]}
            del initial,last,best,optimizer
            gc.collect()
            set_seed(42)
            model=BoneVQAModel.load(str(run/"last.pt"),device="cpu").eval()
            assert model.cfg["use_latent_prompt"] and model.cfg["prior_labels"]==[]
            assert not any(model.cfg[k] for k in ("use_visual_prompt","use_lens","use_textual_prompt","use_region_in_prompt"))
            assert model.cfg["fracture_yes_threshold"]==0.5 and model.cfg.get("fracture_platt_slope") is None
            assert model.cfg["latent_context_control_eval"]==model.cfg["latent_context_control"]
            with torch.enable_grad():
                params=dict(model.named_parameters())
                out=model(collate_fn([train[i] for i in (62,0,3,7)]),stage=3)
                grads=torch.autograd.grad(out["loss_answer"],[params[k] for k in selected],allow_unused=True)
                grad={k:0.0 if v is None else float(v.abs().max()) for k,v in zip(selected,grads)}
                assert all(v>0 for v in grad.values())
                checks[name]["isolated_TRAIN_answer_gradient_max"]=grad
                del out,grads
            conditions={}
            for condition in ("correct_image","component_disjoint_wrong_image","full321_fracture"):
                ds=full if condition=="full321_fracture" else pilot
                chosen=indices if condition=="full321_fracture" else list(range(142))
                rows=[]
                for offset in range(0,len(chosen),4):
                    batch=collate_fn([ds[i] for i in chosen[offset:offset+4]])
                    images=batch["images"] if condition!="component_disjoint_wrong_image" else [pixels[donors[iid]] for iid in batch["image_ids"]]
                    pred=model.predict_batch({"images":images,"questions":batch["questions"]},tra_chi_tiet=True)
                    for i,item in enumerate(pred):
                        row={"image_id":batch["image_ids"][i],"question":batch["questions"][i],"gt":batch["answers"][i],"answer_type":batch["answer_types"][i],"pred":item["answer"]}
                        if "p_yes" in item:
                            row["p_yes"]=item["p_yes"]
                        rows.append(row)
                conditions[condition]={"metrics":metrics(rows),"records":rows}
                state.update(status="running",run_name=name,completed_condition=condition)
                save()
            assert [(r["image_id"],r["gt"]) for r in conditions["full321_fracture"]["records"]]==[(r["image_id"],r["gt"]) for r in bfull["records"]]
            good={r["image_id"]:r for r in conditions["correct_image"]["records"] if r["question"]==canonical}
            wrong={r["image_id"]:r for r in conditions["component_disjoint_wrong_image"]["records"] if r["question"]==canonical}
            error=max(abs(r["p_yes"]-good[donors[iid]]["p_yes"]) for iid,r in wrong.items())
            assert error<=1e-6
            checks[name]["actual_pixel_donor_score_max_error"]=error
            results[name]=conditions
            del model
            gc.collect()
        endpoints={name:r["full321_fracture"]["records"] for name,r in results.items()}
        endpoints["B_ROCO_seed42"]=bfull["records"]
        groups=defaultdict(list)
        for i,row in enumerate(bfull["records"]):
            groups[lookup[row["image_id"]]].append(i)
        keys=sorted(groups)
        rng=random.Random(20261020)
        p_name="P_seed42_target_bounded_v3"
        diffs={name:{"AUROC":[],"AUPRC":[]} for name in endpoints if name!=p_name}
        for _ in range(500):
            take=[i for key in rng.choices(keys,k=len(keys)) for i in groups[key]]
            y=[normalize_answer(bfull["records"][i]["gt"])=="yes" for i in take]
            if len(set(y))<2:
                continue
            values={name:(roc_auc_score(y,[r[i]["p_yes"] for i in take]),average_precision_score(y,[r[i]["p_yes"] for i in take])) for name,r in endpoints.items()}
            for name in diffs:
                for j,key in enumerate(("AUROC","AUPRC")):
                    diffs[name][key].append(values[p_name][j]-values[name][j])
        intervals={name:{k:np.percentile(v,[2.5,97.5]).tolist() for k,v in d.items()} for name,d in diffs.items()}
        report={"scope":state["scope"],"new_results":results,"artifact_and_gradient_checks":checks,"B_reference_full321":bfull["metrics"],"B_reference_QA":old["pilot142_QA"][bname+"/correct_image"]["metrics"],"B_evaluation_sha256":old_state["report_sha256"],"paired_component_bootstrap95_P_minus_controls_descriptive":intervals,"input_keys_only":["images","questions"],"raw_threshold":0.5,"elapsed_seconds":time.perf_counter()-started,"calibration_or_test_used":False,"long_training_released":False,"limitations":["single downstream seed42 on inspected modelvalidation;screening only","PvsB not capacity/FLOP/time matched;all P control initial weights and parameter counts matched","same321images not independent cohorts;descriptive bootstrap not clinical confirmation","all results reported,including negative controls,no threshold or epoch selection"]}
        output.write_text(json.dumps(report,indent=2)+"\n")
        state.update(status="complete_pending_interpretation",finished_utc=datetime.now(timezone.utc).isoformat(),report_sha256=sha(output))
        save()
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry",failure=repr(error))
        save()
        raise


if __name__=="__main__":
    main()
