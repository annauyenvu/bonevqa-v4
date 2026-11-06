"""Frozen-weight, bounded TRAIN-only upstream and image-dependence diagnostic."""
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from src.data import build_dataset, collate_fn
from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.protocol import records_of
from src.train import build_split
from src.utils import compute_metrics, normalize_answer, set_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def main():
    folder = Path("refactor_v2/reports/pmc_alignment_execution_v3")
    output = folder / "bounded_train_upstream_image_diagnostic.json"
    progress = folder / "bounded_train_diagnostic_progress.json"
    if output.exists() or progress.exists():
        raise RuntimeError("Preserve existing diagnostic; do not overwrite or duplicate")
    checks = json.loads((folder / "independent_artifact_verification.json").read_text())
    if checks.get("passed") is not True:
        raise RuntimeError("PMC artifacts must be verified first")
    cfg = json.loads((folder / "released_config.json").read_text())
    data = build_split(build_dataset, "pmc_vqa", "train", None, True, cfg["data"]["loader_kwargs"])
    rows = records_of(data)
    indices, seen = [], set()
    for index, row in enumerate(rows):
        iid = row.get("image_id") or row.get("image") or row.get("image_path")
        if not iid:
            raise RuntimeError("Missing image identity")
        if iid not in seen:
            seen.add(iid)
            indices.append(index)
        if len(indices) == 32:
            break
    if len(indices) != 32 or any(rows[i]["answer_type"] != "OPEN" for i in indices):
        raise RuntimeError("Require32 distinct TRAIN images and generative targets")
    status = {"status": "running", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "32 deterministic distinct PMC TRAIN images; frozen weights; no test or optimization"}
    progress.write_text(json.dumps(status, indent=2) + "\n")
    started = time.perf_counter()
    selected = [rows[i] for i in indices]
    report = {"scope": status["scope"], "selected_indices": indices,
              "selected_records_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True, default=str).encode()).hexdigest(),
              "conditions": {}, "test_used": False, "optimizer_steps": 0,
              "limitations": ["TRAIN subset is not efficacy/generalization evidence", "image donors are distinct images, not a scientific component-matched control",
                              "ROCO and PMC have different training exposure; comparison diagnoses this initialization chain only"],
              "next_stage_released": False}
    paths = {"roco": Path(cfg["init_from"]), "pmc": Path("refactor_v2/checkpoints/revised_pmcvqa_alignment_v3/best.pt")}
    for name, path in paths.items():
        set_seed(42)
        model = BoneVQAModel.load(str(path), device="cpu").eval()
        before = sha(path)
        if any(model.cfg.get(key) for key in ("use_visual_prompt", "use_lens", "use_latent_prompt", "use_textual_prompt", "use_region_in_prompt")):
            raise RuntimeError("Require original image/raw question B path")
        for condition in ("correct_image", "cyclic_wrong_image"):
            records, nll_sum, tokens = [], 0.0, 0
            for offset in range(0, len(indices), 4):
                nb = normalize_batch(collate_fn([data[i] for i in indices[offset:offset + 4]]))
                images = nb["images"]
                if condition == "cyclic_wrong_image":
                    images = images[1:] + images[:1]
                core = model._encode_core(images, nb["questions"], nb["regions"], [None] * 4, model.cfg["prompt_kind"], False)
                prefix = model._llm_prefix(core)
                inputs, attention, labels = model._build_llm_inputs(prefix, core["q_texts"], nb["answers"])
                logits = model.llm(inputs_embeds=inputs, attention_mask=attention).logits
                targets = labels[:, 1:]
                nll_sum += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100, reduction="sum"))
                tokens += int((targets != -100).sum())
                del logits, inputs
                generated = model.generate_open(core, core["q_texts"])
                for i, pred in enumerate(generated):
                    records.append({"image_id": nb["image_ids"][i], "question": nb["questions"][i], "gt": nb["answers"][i],
                                    "pred": pred["text"], "answer_type": "OPEN"})
            report["conditions"][name + "/" + condition] = {"checkpoint_sha256": before, "metrics": compute_metrics(records),
                "teacher_forced_token_ce": nll_sum / tokens, "supervised_tokens": tokens, "records": records}
            status.update(checkpoint=name, condition_completed=condition)
            progress.write_text(json.dumps(status, indent=2) + "\n")
        if sha(path) != before:
            raise RuntimeError("Frozen checkpoint changed")
        del model
        gc.collect()
    for name in paths:
        correct = report["conditions"][name + "/correct_image"]
        wrong = report["conditions"][name + "/cyclic_wrong_image"]
        correct["wrong_minus_correct_token_ce"] = wrong["teacher_forced_token_ce"] - correct["teacher_forced_token_ce"]
        correct["generated_answers_changed_by_image_swap"] = sum(normalize_answer(a["pred"]) != normalize_answer(b["pred"]) for a, b in zip(correct["records"], wrong["records"]))
    report["elapsed_seconds"] = time.perf_counter() - started
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    status.update(status="complete_pending_interpretation", finished_utc=datetime.now(timezone.utc).isoformat(), report_sha256=sha(output))
    progress.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({k: {m: v for m, v in result.items() if m != "records"} for k, result in report["conditions"].items()}, indent=2))


if __name__ == "__main__":
    main()
