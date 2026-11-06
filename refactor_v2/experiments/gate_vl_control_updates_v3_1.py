"""Save fresh pilot initial states,then discard small control wiring updates."""
import copy
import gc
import json
import os
import time
from pathlib import Path

import torch
from src.checkpointing import compatible_state
from src.data import build_dataset,collate_fn
from src.data.fracatlas_vqa import FracAtlasVQADataset
from src.models.bonevqa import BoneVQAModel
from src.train import build_split,seed_training_epoch
from src.utils import set_seed
from experiments.vl_bounded_execution_contract_v3_1 import ARMS,INIT,INIT_SHA,validate_plan
from experiments.vl_bounded_proposal_contract_v3_1 import sha


def hashes(state):
    import hashlib
    return {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in state.items()}


def main():
    root=Path.cwd()
    folder=root/"refactor_v2/reports/vl_control_update_gates_v3_1"
    assert not folder.exists()
    plans=root/"refactor_v2/reports/vl_bounded_execution_contract_v3_1"
    contract=json.loads((plans/"report.json").read_text())
    assert contract["passed_plan_and_caps_tests"]
    controls=root/"refactor_v2/reports/vl_matched_control_audit_v3_1"
    audit=json.loads((controls/"report.json").read_text())
    assert audit["passed"]
    base=json.loads((root/"refactor_v2/reports/fracatlas_scale_pilot_v3/fracatlas_B_roco_scale_pilot_v3.json").read_text())
    loader=copy.deepcopy(base["data"]["loader_kwargs"])
    loader.update(with_masks=False,with_boxes=False,boxes_from=str(root/"refactor_v2/reports/vl_pilot_proposals640_v3_1/boxes.json"),boxes_as_masks=True,roi_control="real")
    ds=build_split(build_dataset,base["data"]["dataset"],"train",None,True,loader)
    real=collate_fn([ds[i] for i in audit["selected_TRAIN_QA_indices"]])
    location=json.loads((controls/"group_translation_location_placebo.json").read_text())
    placebo=dict(real)
    placebo["boxes"]=[location[i] or None for i in real["image_ids"]]
    placebo["masks"]=[FracAtlasVQADataset._boxes_to_masks(location[i],im.size) for i,im in zip(real["image_ids"],real["images"])]
    assert sha(INIT)==INIT_SHA
    initial=torch.load(INIT,map_location="cpu",weights_only=False)
    folder.mkdir()
    state={"status":"running","pid":os.getpid(),"scope":"fourEXACTpilotTRAINQA control-specificwiringupdatesONLY;not qualitytraining"}
    def save():
        (folder/"progress.json").write_text(json.dumps(state,indent=2)+"\n")
    save()
    started=time.perf_counter()
    results={}
    try:
        for arm in ARMS:
            path=plans/(arm+".json")
            assert sha(path)==contract["config_sha256"][arm]
            config=json.loads(path.read_text())
            validate_plan(config)
            set_seed(42)
            model=BoneVQAModel(config["model"],initial["answer_vocab"],device="cpu")
            model.load_state_dict(compatible_state(model.trainable_state_dict(),initial["state"],base["checkpoint_allow_mismatch_prefixes"]),strict=False)
            current=model.trainable_state_dict()
            assert all(torch.equal(current[k],v) for k,v in initial["state"].items())
            assert hashes(current)==audit["initial_tensor_hashes"][arm[0]]
            fresh_path=folder/(arm+"_PREtask_initialization.pt")
            model.save(str(fresh_path),{"scope":"freshPREtaskinitialization;zeroQAupdates","parent_ROCO_PREtask_sha256":INIT_SHA,"arm":arm})
            fresh_hash=sha(fresh_path)
            batch=placebo if arm.endswith("location_placebo") else real
            params=dict(model.named_parameters())
            if arm.startswith("V_"):
                names=("prompt_mixer.gamma_visual","prompt_mixer.visual_proj.0.weight")
            else:
                names=("lens_fusion.view_emb",next(k for k in params if k.startswith("visual.") and "lora_B" in k))
            observations=[]
            for phase in config["phases"]:
                stage=phase["stage"]
                before={k:params[k].detach().clone() for k in names}
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,betas=(0.9,0.98),weight_decay=0.01)
                grads_by_step=[]
                for step in range(2):
                    seed_training_epoch(42,phase["global_epoch_offset"]+step)
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    out=model(batch,stage=stage)
                    objective=out["loss_answer"] if arm.startswith("V_") else out["loss_lens_alignment"]
                    gradients=torch.autograd.grad(objective,[params[k] for k in names],retain_graph=True,allow_unused=True)
                    values={k:0.0 if v is None else float(v.abs().max()) for k,v in zip(names,gradients)}
                    assert torch.isfinite(out["loss"])
                    out["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    optimizer.step()
                    grads_by_step.append(values)
                updates={k:float((params[k].detach()-before[k]).abs().max()) for k in names}
                assert all(v>0 for v in grads_by_step[-1].values()) and all(v>0 for v in updates.values())
                observations.append({"stage":stage,"actual_smoke_updates":2,"isolated_objective":"answer_only" if arm.startswith("V_") else "lens_alignment_only","gradient_max":grads_by_step,"selected_actual_updates":updates})
                del optimizer,out,gradients,before
            assert sha(fresh_path)==fresh_hash
            results[arm]={"passed":True,"config_sha256":sha(path),"fresh_initialization_path":str(fresh_path),"fresh_initialization_sha256":fresh_hash,"all_initial_tensors_match_frozen_control_audit":len(current),"shared_B_tensors_exact":263,"observations":observations,"smoke_updated_weights_discarded":True}
            (folder/(arm+"_gate.json")).write_text(json.dumps(results[arm],indent=2)+"\n")
            state.update(completed_arm=arm);save()
            del model,current,params
            gc.collect()
        report={"passed":True,"scope":state["scope"],"results":results,"selected_TRAIN_QA_indices":audit["selected_TRAIN_QA_indices"],"elapsed_seconds":time.perf_counter()-started,"quality_training_released":False,"calibration_or_test_used":False,"pending":"independentfreshinitialstateverificationanddedicatedboundedexecutorsealbeforefirstqualityupdate","limitations":["2updatesperVarm,4perLarmfortwophases;technicalonlynot296stepqualitypilot","Lisolatedalignmentrouteproofnotanswer-onlylensgradientbecauselensisauxiliaryteacher","allupdatedsmokeweightsdiscarded,noqualitycheckpointcreated"]}
        (folder/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        state.update(status="passed_technical_controls_only",report_sha256=sha(folder/"report.json"));save()
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry",failure=repr(error));save();raise


if __name__=="__main__":
    main()
