"""Fresh-process V/L reconstruction and public inference contract;TRAIN only."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from src.data import build_dataset,collate_fn
from src.models.bonevqa import BoneVQAModel,normalize_batch
from src.train import build_split
from src.utils import set_seed
from experiments.validate_detector640_TRAIN_fixture_v3 import validate as validate_fixture

ROOT=Path.cwd()
FOLDER=ROOT/"refactor_v2/reports/vl_detector640_independent_v3"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def worker(variant,seed):
    output=FOLDER/(variant+"_seed"+str(seed)+".pt")
    assert not output.exists()
    gate=json.loads((ROOT/"refactor_v2/reports/vl_detector640_target_routes_v3.json").read_text())
    proposals=ROOT/gate["proposals"]
    validate_fixture(proposals)
    assert sha(proposals)==gate["proposal_sha256"]
    base=json.loads((ROOT/"refactor_v2/configs/revised_sealed_candidates_v3/b.json").read_text())
    loader=copy.deepcopy(base["data"]["loader_kwargs"])
    loader.update(with_masks=False,with_boxes=False,boxes_from=str(proposals),boxes_as_masks=True)
    ds=build_split(build_dataset,base["data"]["dataset"],"train",None,True,loader)
    batch=collate_fn([ds[i] for i in gate["train_indices"]])
    nb=normalize_batch(batch)
    checkpoint=ROOT/("refactor_v2/checkpoints/gate_"+variant.lower()+"_detector640_target_route_v3/smoke.pt")
    assert sha(checkpoint)==json.loads((FOLDER/"release.json").read_text())["checkpoint_sha256"][variant]
    set_seed(seed)
    model=BoneVQAModel.load(str(checkpoint),device="cpu").eval()
    assert not model.cfg["use_region_in_prompt"] and not model.cfg["use_latent_prompt"]
    before={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}
    def core(indices,use_boxes):
        images=[nb["images"][i] for i in indices]
        questions=[nb["questions"][i] for i in indices]
        boxes=[nb["boxes"][i] for i in indices] if use_boxes else [None]*len(indices)
        masks=model._prepare_masks(images,[None]*len(indices),boxes,[nb["image_ids"][i] for i in indices])
        out=model._encode_core(images,questions,[None]*len(indices),masks,model.cfg["prompt_kind"],False)
        return {"closed":out["logits_closed"].cpu(),"fracture":out["fracture_logit"].cpu(),"type":out["type_logit"].cpu(),"prefix":model._llm_prefix(out).cpu()}
    proper=core(list(range(4)),True)
    empty=core(list(range(4)),False)
    single=core([0],True)
    reverse=core([3,2,1,0],True)
    errors={"single":{k:float((v[:1]-single[k]).abs().max()) for k,v in proper.items()},"reverse":{k:float((v-reverse[k].flip(0)).abs().max()) for k,v in proper.items()},"noROI":{k:float((v-empty[k]).abs().max()) for k,v in proper.items()}}
    assert max(errors["single"].values())<=1e-4 and max(errors["reverse"].values())<=1e-4
    args={"images":nb["images"],"questions":nb["questions"]}
    if variant=="V":
        args["boxes"]=nb["boxes"]
        assert errors["noROI"]["closed"]>0 and errors["noROI"]["fracture"]>0 and errors["noROI"]["prefix"]>0
    else:
        assert all(v==0 for v in errors["noROI"].values())
    prediction=model.predict_batch(args,tra_chi_tiet=True)
    singles={"images":[nb["images"][0]],"questions":[nb["questions"][0]]}
    if variant=="V":
        singles["boxes"]=[nb["boxes"][0]]
    one=model.predict_batch(singles,tra_chi_tiet=True)
    assert prediction[0]["answer"]==one[0]["answer"]
    if variant=="L":
        withROI=model.predict_batch({**args,"boxes":nb["boxes"]},tra_chi_tiet=True)
        assert [p["answer"] for p in prediction]==[p["answer"] for p in withROI]
    assert before=={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}
    torch.save({"proper":proper,"noROI":empty,"errors":errors,"answers":[p["answer"] for p in prediction],"p_yes":[p.get("p_yes") for p in prediction],"all_weights_unchanged":True,"weights_hashes":before,"input_keys_only":list(args)},output)


def main():
    assert not FOLDER.exists()
    commands=subprocess.check_output(["ps","-axo","command"],text=True)
    if any(" -m src.train " in row for row in commands.splitlines()):
        raise RuntimeError("Active training;do not compete")
    source=ROOT/"refactor_v2/reports/vl_detector640_target_routes_v3.json"
    assert json.loads(source.read_text())["all_passed"]
    FOLDER.mkdir()
    state={"status":"running","pid":os.getpid(),"scope":"fourTRAINQArows;twofreshprocesses/checkpoint;no quality training"}
    def save():
        (FOLDER/"progress.json").write_text(json.dumps(state,indent=2)+"\n")
    save()
    started=time.perf_counter()
    try:
        release={"source_gate_sha256":sha(source),"checkpoint_sha256":{v:sha(ROOT/("refactor_v2/checkpoints/gate_"+v.lower()+"_detector640_target_route_v3/smoke.pt")) for v in ("V","L")},"worker_source_sha256":sha(__file__),"seeds":[42,90210],"public_inference_no_GT_type_region_or_annotation_ROI":True}
        (FOLDER/"release.json").write_text(json.dumps(release,indent=2)+"\n")
        checks={}
        for variant in ("V","L"):
            for seed in (42,90210):
                subprocess.run([sys.executable,__file__,"--worker",variant,str(seed)],check=True)
                state.update(completed_variant=variant,completed_seed=seed)
                save()
            a,b=[torch.load(FOLDER/(variant+"_seed"+str(s)+".pt"),map_location="cpu",weights_only=False) for s in (42,90210)]
            differences={k:float((v-b["proper"][k]).abs().max()) for k,v in a["proper"].items()}
            assert all(v==0 for v in differences.values()) and a["answers"]==b["answers"] and a["p_yes"]==b["p_yes"] and a["weights_hashes"]==b["weights_hashes"]
            checks[variant]={"fresh_process_endpoint_max_differences":differences,"public_answers_scores_exact":True,"reconstruction_seed_independent":True,"batch_and_noROI_differences":a["errors"],"public_input_keys":a["input_keys_only"],"all_trainable_weights_unchanged":True}
        report={"passed":True,"scope":state["scope"],"checks":checks,"elapsed_seconds":time.perf_counter()-started,"source_gate_sha256":release["source_gate_sha256"],"checkpoint_sha256":release["checkpoint_sha256"],"calibration_or_test_used":False,"scientific_VL_training_or_production_released":False,"limitations":["two-updates/fourTRAINrows wiringonly,not efficacy or exhaustive batch invariance","Vboxesareautomaticdetectorfixtureoutput,notannotationROI","Loriginalimagequestiononly;ROIteacheronlyduringtraining","incomplete8TRAINproposalsstillrejectedbyscientificCLI;rolecontractcontrols/qualityrunsstillpending"]}
        (FOLDER/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        state.update(status="passed_technical_only",report_sha256=sha(FOLDER/"report.json"))
        save()
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry",failure=repr(error))
        save()
        raise


if __name__=="__main__":
    if len(sys.argv)>1:
        worker(sys.argv[2],int(sys.argv[3]))
    else:
        main()
