"""Release exactly three seed42 capped GM+LF/control arms; no long-run chain."""
import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from src.checkpointing import compatible_state
from src.context_controls import ComponentBatchSampler, component_disjoint_donors, load_component_lookup
from src.data import build_dataset, collate_fn
from src.models.bonevqa import BoneVQAModel
from src.protocol import manifest, records_of
from src.train import build_answer_vocab, build_split, collect_answers, seed_training_epoch
from src.utils import set_seed
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def hashes(state):
    return {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in state.items()}


def main():
    root=Path.cwd()
    folder=root/"refactor_v2/reports/P_target_semantic_pilot_v3"
    technical=root/"refactor_v2/reports/P_target_scale_gate_v3"
    parent=root/"refactor_v2/reports/fracatlas_scale_pilot_v3"
    if folder.exists():
        raise RuntimeError("Existing pilot;inspect,never duplicate or overwrite")
    commands=subprocess.check_output(["ps","-axo","command"],text=True)
    if any(" -m src.train " in row for row in commands.splitlines()):
        raise RuntimeError("Existing training alive")
    expected_env={"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","OMP_NUM_THREADS":"8","HF_HOME":str(root/".evaluation_runtime/hf_cache"),"BONEVQA_DATA":str(root/".evaluation_runtime/data")}
    assert all(os.environ.get(k)==v for k,v in expected_env.items())
    subprocess.run([sys.executable,"refactor_v2/scripts/seal_technical_reference_v3.py","--verify"],check=True)
    verify=json.loads((technical/"independent_verification.json").read_text())
    verified=json.loads((technical/"verification_progress.json").read_text())
    assert verify["passed"] and sha(technical/"independent_verification.json")==verified["report_sha256"]
    gate=json.loads((technical/"report.json").read_text())
    assert gate["passed"] and sha(technical/"technical_only.pt")==verify["reload_checkpoint_sha256"]
    protocol=json.loads((technical/"next_bounded_protocol.json").read_text())
    assert protocol["status"]=="prepared_NOT_execution_release" and not protocol["calibration_or_test_allowed"]
    initial_path=parent/"fracatlas_B_roco_scale_pilot_v3_initialization.pt"
    assert sha(initial_path)==verify["PREtask_initialization_sha256"]
    initial=torch.load(initial_path,map_location="cpu",weights_only=False)
    reference=json.loads((parent/"execution_release.json").read_text())
    assert hashes(initial["state"])==reference["initial_tensor_hashes"]["fracatlas_B_roco_scale_pilot_v3"]
    base=json.loads((parent/"fracatlas_B_roco_scale_pilot_v3.json").read_text())
    assert base["seed"]==42 and base["train"]["epochs"]==4 and base["train"]["stage"]==3
    loader=base["data"]["loader_kwargs"]
    assert loader["sealed_records_by_split"]["calibration"] is None and loader["sealed_records_by_split"]["test"] is None
    datasets={role:build_split(build_dataset,base["data"]["dataset"],role,None,True,loader) for role in ("train","model_val")}
    actual={role:manifest(ds) for role,ds in datasets.items()}
    assert actual==reference["actual_manifests"] and len(datasets["train"])==1173 and len(datasets["model_val"])==142
    vocab=build_answer_vocab(collect_answers(datasets["train"]),base["data"]["answer_vocab_size"])
    assert vocab==initial["answer_vocab"]==json.loads((parent/"TRAIN_answer_vocab.json").read_text())
    batch_cfg=base["train"]["component_batching"]
    lookup,_=load_component_lookup(batch_cfg["manifest"],batch_cfg["seal"],root)
    sampler=ComponentBatchSampler([lookup[r["image_id"]] for r in records_of(datasets["train"])],4,42)
    schedules=[]
    donor_audits=[]
    for epoch in range(4):
        sampler.set_epoch(epoch)
        batches=list(sampler)
        assert len(batches)==294 and sorted(i for batch in batches for i in batch)==list(range(1173))
        # The frozen policy permits repeated components up to half a batch,
        # but never a donor from the recipient's component. Do not alter B order.
        repeated=0
        for batch in batches:
            groups=[lookup[records_of(datasets["train"])[i]["image_id"]] for i in batch]
            donors=component_disjoint_donors(groups)
            assert sorted(donors)==list(range(len(batch))) and all(groups[i]!=groups[j] for i,j in enumerate(donors))
            repeated+=int(len(set(groups))<len(groups))
        donor_audits.append({"epoch":epoch,"batches":len(batches),"batches_with_repeated_components":repeated,"every_row_has_bijective_cross_component_donor":True})
        schedules.append(hashlib.sha256(json.dumps(batches).encode()).hexdigest())
    folder.mkdir()
    state={"status":"preparing_and_gating","driver_pid":os.getpid(),"started_utc":now(),"scope":"three seed42 bounded GM+LF semantic-control arms only;not efficacy or long training"}
    def save():
        (folder/"progress.json").write_text(json.dumps(state,indent=2)+"\n")
    save()
    try:
        (folder/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
        configs,initial_hashes,smokes,counts=[],{}, {},{}
        shared_P=None
        for kind,mode,context in (("P","train","conditioned"),("P_random_bank","random","conditioned"),("P_unconditioned","train","unconditioned")):
            name=kind+"_seed42_target_bounded_v3"
            assert not (root/"refactor_v2/checkpoints"/name).exists()
            model_cfg=copy.deepcopy(initial["config"])
            model_cfg.update(prompt_variant="P",use_latent_prompt=True,prior_labels=[],latent_answer_bank_mode=mode,latent_context_control=context,latent_context_control_eval=context,grad_checkpoint=False,llm_grad_checkpoint=False)
            set_seed(42)
            model=BoneVQAModel(model_cfg,vocab,device="cpu")
            transferred=compatible_state(model.trainable_state_dict(),initial["state"],base["checkpoint_allow_mismatch_prefixes"])
            model.load_state_dict(transferred,strict=False)
            tensors=model.trainable_state_dict()
            assert finite_tree(tensors) and all(torch.equal(tensors[k],v) for k,v in initial["state"].items())
            initial_hashes[name]=hashes(tensors)
            if shared_P is None:
                shared_P=initial_hashes[name]
            else:
                assert shared_P==initial_hashes[name]
            counts[name]=sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert counts[name]==99957064
            init_path=folder/(name+"_initialization.pt")
            model.save(str(init_path),{"scope":"fresh PRE-task target P/control initialization;no task training","ROCO_PREtask_sha256":sha(initial_path)})
            # Two discarded TRAIN-only updates verify each control can carry answer loss.
            batch=collate_fn([datasets["train"][i] for i in gate["selected_TRAIN_indices"]])
            params=dict(model.named_parameters())
            names=("fusion.gamma_latent","latent.latent")
            before={k:params[k].detach().clone() for k in names}
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,betas=(0.9,0.98),weight_decay=0.01)
            observations=[]
            for step in range(2):
                seed_training_epoch(42,step)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                out=model(batch,stage=3)
                gradients=torch.autograd.grad(out["loss_answer"],[params[k] for k in names],retain_graph=True,allow_unused=True)
                grad={k:0.0 if v is None else float(v.abs().max()) for k,v in zip(names,gradients)}
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                observations.append(grad)
            updates={k:float((params[k].detach()-before[k]).abs().max()) for k in names}
            assert all(v>0 for v in observations[-1].values()) and all(v>0 for v in updates.values()) and finite_tree(model.trainable_state_dict())
            smokes[name]={"answer_gradients":observations,"actual_updates":updates,"discarded_training_not_used_for_init":True}
            cfg=copy.deepcopy(base)
            cfg.update(run_name=name,init_from=str(init_path),recipe_status="released_three_arm_seed42_bounded_P_only",allow_random_init=False,figure_dir=str(folder/name/"figures"))
            cfg["model"]=model_cfg
            path=folder/(name+".json")
            path.write_text(json.dumps(cfg,indent=2)+"\n")
            configs.append((cfg,path))
            del model,tensors,transferred,optimizer,params,before,out,gradients
            gc.collect()
        release={"passed":True,"scope":state["scope"],"protocol_sha256":sha(folder/"protocol.json"),"technical_verification_sha256":verified["report_sha256"],"source_PREtask_initialization_sha256":sha(initial_path),"data_seal_sha256":sha(loader["seal_path"]),"TRAIN_vocab_sha256":sha(parent/"TRAIN_answer_vocab.json"),"actual_manifests":actual,"sampler_all4epochs_no_row_loss":True,"epoch_batch_schedule_sha256":schedules,"initial_tensor_hashes":initial_hashes,"initialization_sha256":{c["run_name"]:sha(c["init_from"]) for c,_ in configs},"config_sha256":{c["run_name"]:sha(p) for c,p in configs},"TRAIN_control_smokes":smokes,"trainable_parameter_counts":counts,"P_controls_parameter_matched":True,"B_parameter_or_compute_matched":False,"budget_each":"4epochs296steps,1173TRAIN142modelvalQA;LASTprimary;CPU8float32","test_or_calibration_used":False,"long_training_released":False}
        release["component_donor_audit"]=donor_audits
        release["source_script_sha256"]=sha(__file__)
        (folder/"execution_release.json").write_text(json.dumps(release,indent=2)+"\n")
        del initial,datasets
        gc.collect()
        for cfg,path in configs:
            assert sha(path)==release["config_sha256"][cfg["run_name"]] and sha(cfg["init_from"])==release["initialization_sha256"][cfg["run_name"]]
            started=time.perf_counter()
            with (folder/(cfg["run_name"]+".log")).open("w") as log:
                child=subprocess.Popen([sys.executable,"-m","src.train","--config",str(path),"--device","cpu"],env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                guard=subprocess.Popen(["caffeinate","-i","-w",str(child.pid)])
                state.update(status="running",run_name=cfg["run_name"],training_pid=child.pid,guard_pid=guard.pid,run_started_utc=now())
                save()
                code=child.wait()
            if code:
                raise RuntimeError(f"{cfg['run_name']} exit{code};preserve,no retry")
            run=root/cfg["checkpoint_dir"]/cfg["run_name"]
            last=torch.load(run/"last.pt",map_location="cpu",weights_only=False)
            opt=torch.load(run/"optimizer.pt",map_location="cpu",weights_only=False)
            history=json.loads((run/"history.json").read_text())
            assert len(history)==4 and last["extra"]["global_step"]==opt["global_step"]==296
            assert finite_tree(last["state"]) and finite_tree(opt)
            assert all(r["val_n"]==142 and r["skipped_no_supervision_batches"]==0 for r in history)
            done={"complete":True,"wall_seconds":time.perf_counter()-started,"history":history,"optimizer_steps":296,"last_sha256":sha(run/"last.pt"),"best_sha256":sha(run/"best.pt"),"quality_gate_passed":False}
            (folder/(cfg["run_name"]+"_completion.json")).write_text(json.dumps(done,indent=2)+"\n")
            del last,opt
            gc.collect()
            state.setdefault("completed_runs",[]).append(cfg["run_name"])
            save()
        state.update(status="three_arms_complete_pending_independent_evaluation",finished_utc=now(),long_training_released=False)
        save()
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry",failure=repr(error),finished_utc=now())
        save()
        raise


if __name__=="__main__":
    main()
