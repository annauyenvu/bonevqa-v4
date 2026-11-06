"""Frozen learned P: TRAIN-only gradient attribution and gate intervention."""
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from src.data import build_dataset, collate_fn
from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.train import build_split
from src.utils import set_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path.cwd()
    folder=root/"refactor_v2/reports/P_target_semantic_pilot_v3"
    output=folder/"learned_strength_TRAIN_only.json"
    assert not output.exists()
    evaluation=json.loads((folder/"independent_evaluation.json").read_text())
    assert sha(folder/"independent_evaluation.json")==json.loads((folder/"independent_evaluation_progress.json").read_text())["report_sha256"]
    started=time.perf_counter()
    reports={}
    for name in evaluation["new_results"]:
        cfg=json.loads((folder/(name+".json")).read_text())
        ds=build_split(build_dataset,cfg["data"]["dataset"],"train",None,True,cfg["data"]["loader_kwargs"])
        batch=collate_fn([ds[i] for i in (62,0,3,7)])
        nb=normalize_batch(batch)
        checkpoint=root/"refactor_v2/checkpoints"/name/"last.pt"
        assert sha(checkpoint)==json.loads((folder/(name+"_completion.json")).read_text())["last_sha256"]
        set_seed(42)
        model=BoneVQAModel.load(str(checkpoint),device="cpu").eval()
        names=("fusion.gamma_latent","latent.latent","fusion.lp_ca_l.norm_q.weight","fusion.lp_ca_i.norm_q.weight","fusion.lp_ca_mm.norm_q.weight")
        params=dict(model.named_parameters())
        before={k:v.detach().cpu().clone() for k,v in model.trainable_state_dict().items()}
        out=model(batch,stage=3)
        # Forward exposes detached scalar loss_cs for logging, not attribution.
        # Reconstruct the same raw TRAIN-only term without modifying frozen code.
        core=model._encode_core(nb["images"],nb["questions"],nb["regions"],[None]*4,model.cfg["prompt_kind"],False)
        target=model.text.encode_frozen(nb["answers"],model.device)
        consistency=model.latent.consistency_loss(core["x_lp"],target["tokens"],target["mask"])
        assert torch.allclose(consistency.detach(),out["loss_cs"],atol=1e-6,rtol=1e-6)
        attribution={}
        for label,loss in (("answer",out["loss_answer"]),("weighted_GM_consistency",model.cfg["eta"]*consistency),("total",out["loss"])):
            grads=torch.autograd.grad(loss,[params[k] for k in names],retain_graph=True,allow_unused=True)
            attribution[label]={k:0.0 if v is None else float(v.abs().max()) for k,v in zip(names,grads)}
        del out,grads,core,target,consistency
        @torch.no_grad()
        def endpoints():
            core=model._encode_core(nb["images"],nb["questions"],[None]*4,[None]*4,model.cfg["prompt_kind"],False)
            prefix=model._llm_prefix(core)
            embeds,attention,labels=model._build_llm_inputs(prefix[2:],core["q_texts"][2:],nb["answers"][2:])
            logits=model.llm(inputs_embeds=embeds,attention_mask=attention).logits
            return {"closed":core["logits_closed"].cpu(),"fracture":core["fracture_logit"].cpu(),"open_answer_logits":logits[:,:-1][labels[:,1:]!=-100].float().cpu(),"latent_delta_mean_norm":float(core["fus"]["latent_delta"].norm(dim=-1).mean())}
        normal=endpoints()
        model.cfg["latent_force_zero"]=True
        zero=endpoints()
        model.cfg["latent_force_zero"]=False
        differences={k:float((normal[k]-zero[k]).abs().max()) for k in ("closed","fracture","open_answer_logits")}
        with torch.no_grad():
            pred=model.predict_batch({"images":nb["images"],"questions":nb["questions"]},tra_chi_tiet=True)
            model.cfg["latent_force_zero"]=True
            neutral=model.predict_batch({"images":nb["images"],"questions":nb["questions"]},tra_chi_tiet=True)
            model.cfg["latent_force_zero"]=False
        assert all(torch.equal(before[k],v) for k,v in model.trainable_state_dict().items())
        reports[name]={"checkpoint_sha256":sha(checkpoint),"learned_gate":float(params["fusion.gamma_latent"].detach()),"isolated_loss_gradient_max":attribution,"gate_zero_endpoint_max_differences":differences,"latent_delta_mean_norm":normal["latent_delta_mean_norm"],"generated_answer_changes_under_gate_zero":sum(a["answer"]!=b["answer"] for a,b in zip(pred,neutral)),"n_TRAIN_rows":4,"all_trainable_weights_unchanged":True}
        del model,params,before
        gc.collect()
    report={"scope":"Frozen-checkpoint TRAIN-only diagnostic;no optimization or efficacy/clinical claim","TRAIN_indices":[62,0,3,7],"arms":reports,"elapsed_seconds":time.perf_counter()-started,"calibration_validation_test_used":False,"limitations":["max-absolute gradient on4rows is not global gradient norm or comprehensive module contribution","OPEN teacher-forced TRAIN answer logits use existing GT only as diagnostic target;public prediction receives onlyimages/questions","nonzero influence is not superiority over semantic controls"],"training_released":False}
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
