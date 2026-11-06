"""Router threshold sensitivity (type head > 0.7) and T token budget, on saved checkpoints.  CPU, read-only."""
import json, sys
from pathlib import Path
import numpy as np, torch

R2 = Path("/Users/uyen/Downloads/BONEVQA/refactor_v2")
sys.path.insert(0, str(R2))
from experiments.route_audit_dyn_v1 import load_arm
from experiments.v5 import study_v5 as S
from src.models.bonevqa import is_closed_question, normalize_answer, is_polar_question

torch.set_num_threads(4)
out = {}
for arm in ("B", "V4", "VLPT"):
    model, spec, _, saved = load_arm(arm)
    ds, collate = S.batcher("0", spec)
    recs = ds["model_val"].records
    pred = json.load(open(saved / "model_val_predictions.json"))["records"]
    free = [i for i, r in enumerate(recs) if not is_closed_question(r["question"])]
    tp, top = {}, {}
    with torch.no_grad():
        for off in range(0, len(free), 16):
            ids = free[off: off + 16]
            b = collate([ds["model_val"][i] for i in ids], role="model_val")
            pb = S.public(b, spec)
            ml = model._prepare_masks(b["images"], pb.get("masks"), pb.get("boxes"), b["image_ids"])
            core = model._encode_core(b["images"], b["questions"], [None] * len(ids), ml, model.cfg["prompt_kind"], False)
            for k, i in zip(range(len(ids)), ids):
                tp[i] = float(torch.sigmoid(core["type_logit"][k].float()))
                top[i] = model.answer_vocab[int(core["logits_closed"][k].argmax())]
    p = np.array([tp[i] for i in free])
    ok_open = np.array([normalize_answer(pred[i]["pred"]) == normalize_answer(recs[i]["answer"]) for i in free], float)
    ok_closed = np.array([normalize_answer(top[i]) == normalize_answer(recs[i]["answer"]) for i in free], float)
    routed_closed_now = p > 0.7
    grid = {}
    for t in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        c = p > t
        unknown = int((~c & routed_closed_now).sum())  # would need a generated answer that was never produced
        grid[str(t)] = dict(routed_closed=int(c.sum()), routed_open=int((~c).sum()), changed_vs_0p7=int((c != routed_closed_now).sum()), unknown_rows=unknown,
                            acc_known=float((np.where(c, ok_closed, ok_open))[~(~c & routed_closed_now)].mean()))
    out[arm] = dict(rows=len(recs), rows_rule_closed=len(recs) - len(free), rows_free_router=len(free), type_prob_quantiles=[float(x) for x in np.percentile(p, [0, 1, 5, 50, 95, 99, 100])], grid=grid,
                    gt_type_counts={a: int(sum(recs[i]["answer_type"] == a for i in free)) for a in ("OPEN", "CLOSED")})
    print(arm, json.dumps(out[arm]), flush=True)
    if arm == "B":
        T = S.arm_spec("T")
        qs = sorted({r["question"] for role in ("train", "model_val", "test") for r in ds[role].records})
        tpl = "This is a bone X-ray image. Answer the question briefly. "
        text = [len(model.text.tokenizer(tpl + q)["input_ids"]) for q in qs]
        llm = [len(model.llm_tokenizer("Question: " + tpl + q + " Answer:", add_special_tokens=False)["input_ids"]) for q in qs]
        tn = len(model.text.tokenizer(tpl.strip(), add_special_tokens=False)["input_ids"])
        out["token_budget"] = dict(n_unique_questions=len(qs), template_tokens=tn, max_text_tokens=max(text), text_limit=model.cfg["max_question_tokens"], truncated_text=sum(x > model.cfg["max_question_tokens"] for x in text),
                                   max_llm_tokens=max(llm), llm_limit=model.cfg["max_question_tokens"] + 8, truncated_llm=sum(x > model.cfg["max_question_tokens"] + 8 for x in llm))
        print(out["token_budget"], flush=True)
    del model
json.dump(out, open(R2 / "reports/route_audit_v1/router.json", "w"), indent=1)
