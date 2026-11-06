"""Answer-only gradients and component routing of the bounded shuffled-P checkpoint."""
import json
import time
from pathlib import Path
import torch
from src.context_controls import ComponentBatchSampler, component_disjoint_donors, load_component_lookup
from src.data import build_dataset, collate_fn
from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.protocol import records_of
from src.train import build_split, seed_training_epoch


def main():
    start = time.perf_counter()
    folder = Path("refactor_v2/checkpoints/component_sampler_resume_train_only_v3_p_shuffled/continuous")
    output = Path("refactor_v2/reports/p_component_answer_route_v3.json")
    if output.exists():
        raise RuntimeError("Existing P answer audit must not be overwritten")
    cfg = json.loads((folder / "config.json").read_text())
    batching = cfg["train"]["component_batching"]
    lookup, _ = load_component_lookup(batching["manifest"], batching["seal"], cfg["data"]["loader_kwargs"]["project_root"])
    ds = build_split(build_dataset, cfg["data"]["dataset"], "train", None, True, cfg["data"]["loader_kwargs"])
    groups = [lookup[row["image_id"]] for row in records_of(ds)]
    sampler = ComponentBatchSampler(groups, cfg["train"]["batch_size"], cfg["seed"])
    indices = next(iter(sampler))
    batch = collate_fn([ds[i] for i in indices])
    batch["component_ids"] = [groups[i] for i in indices]
    expected = batch["component_ids"]
    donors = component_disjoint_donors(expected)
    model = BoneVQAModel.load(str(folder / "last.pt"), device="cpu")
    captured = []
    def record_groups(module, args, kwargs):
        captured.append(kwargs.get("context_group_ids"))
    handle = model.fusion.register_forward_pre_hook(record_groups, with_kwargs=True)
    seed_training_epoch(42, 2)
    model.train()
    out = model(batch, stage=3)
    names = ("fusion.gamma_latent", "latent.latent", "fusion.lp_ca_l.norm_q.weight",
             "fusion.lp_ca_i.norm_q.weight", "fusion.lp_ca_mm.norm_q.weight")
    parameters = dict(model.named_parameters())
    gradients = torch.autograd.grad(out["loss_answer"], [parameters[name] for name in names], allow_unused=True)
    maxima = {name: 0.0 if gradient is None else float(gradient.abs().max()) for name, gradient in zip(names, gradients)}
    finite = all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients)
    nb = normalize_batch(batch)
    def endpoints(zero):
        model.cfg["latent_force_zero"] = zero
        model.eval()
        with torch.no_grad():
            core = model._encode_core(nb["images"], nb["questions"], nb["regions"], [None] * len(indices),
                                      model.cfg["prompt_kind"], False, context_group_ids=expected)
        return core["logits_closed"], core["fracture_logit"]
    real, zero = endpoints(False), endpoints(True)
    model.cfg["latent_force_zero"] = False
    handle.remove()
    qc = json.loads((folder / "prompt_module_qc.json").read_text())["P"]
    differences = {key: float((a - b).abs().max()) for key, a, b in zip(("closed", "fracture"), real, zero)}
    report = {"scope": "4 CLOSED outer-TRAIN rows from bounded shuffled-P smoke; no OPEN or quality evidence",
        "selected_indices": indices, "component_ids": expected, "donors": donors,
        "all_fusion_calls_receive_exact_components": all(value == expected for value in captured),
        "answer_only_gradients": maxima, "gradients_finite": finite,
        "gate_zero_endpoint_differences": differences,
        "training_total_loss_qc_updated_tensor_count": len(qc["updated_tensors"]),
        "seconds": time.perf_counter() - start,
        "passed": finite and all(value > 0 for value in maxima.values()) and bool(qc["updated_tensors"])
            and all(value == expected for value in captured) and all(value > 0 for value in differences.values())}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
