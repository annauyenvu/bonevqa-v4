"""Prospective bounded data-expansion candidate; does not authorize or start training."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from src.context_controls import ComponentBatchSampler, load_component_lookup
from src.data import build_dataset
from src.data.sealed_manifest import load_sealed_records
from src.protocol import assert_disjoint_splits, manifest, records_of
from src.train import build_split


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path.cwd()
    output = root / "refactor_v2/reports/fracatlas_scale_pilot_v3"
    if output.exists():
        raise RuntimeError("Existing scale candidate must be preserved")
    old = root / "refactor_v2/reports/fracatlas_baseline_pilot_v3"
    base = json.loads((root / "refactor_v2/configs/revised_sealed_candidates_v3/b.json").read_text())
    loader = base["data"]["loader_kwargs"]
    batching = base["train"]["component_batching"]
    lookup, component_hash = load_component_lookup(batching["manifest"], batching["seal"], root)
    source = load_sealed_records(loader["sealed_records_by_split"]["train"], loader["seal_path"], root)
    groups = defaultdict(list)
    for row in source:
        groups[lookup[row["image_id"]]].append(row)
    strata, mixed = {0: [], 1: []}, []
    for key, rows in groups.items():
        labels = {int(row["fractured"]) for row in rows}
        if len(labels) != 1:
            mixed.append(key)
        else:
            strata[next(iter(labels))].append(key)
    def rank(key):
        return hashlib.sha256(f"target_B_pilot_seed42_component{key}".encode()).hexdigest()
    chosen = set(sorted(strata[0], key=rank)[:128] + sorted(strata[1], key=rank)[:128])
    assert len(chosen) == 256
    train = [row for row in source if lookup[row["image_id"]] in chosen]
    parent_seal = json.loads((old / "data_seal.json").read_text())
    assert set(parent_seal["selected_components"]["train"]) <= chosen
    val = load_sealed_records(old / "model_val.json", old / "data_seal.json", root)
    val_groups = {lookup[row["image_id"]] for row in val}
    assert not chosen & val_groups
    assert len(train) <= 2000 and len(val) == 142
    output.mkdir()
    artifacts = {Path(batching["manifest"]).relative_to(root).as_posix(): component_hash}
    for name, rows in (("train", train), ("model_val", val)):
        path = output / (name + ".json")
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        artifacts[path.relative_to(root).as_posix()] = sha(path)
    seal_path = output / "data_seal.json"
    seal_path.write_text(json.dumps({"status": "sealed_development_data", "scope": "bounded_target_B_data_expansion_candidate_only",
        "parent_TRAIN_sha256": sha(loader["sealed_records_by_split"]["train"]), "parent_seal_sha256": sha(loader["seal_path"]),
        "artifacts": artifacts, "selected_TRAIN_components": sorted(chosen), "selected_model_val_components": sorted(val_groups),
        "mixed_label_TRAIN_components_not_used_in_candidate": mixed, "original_data_unchanged": True,
        "calibration_or_test_read": False, "training_released": False}, indent=2) + "\n")
    local = dict(loader)
    local.update(seal_path=str(seal_path), sealed_records_by_split={"train": str(output / "train.json"),
                 "model_val": str(output / "model_val.json"), "calibration": None, "test": None})
    datasets = {role: build_split(build_dataset, base["data"]["dataset"], role, None, True, local) for role in ("train", "model_val")}
    assert_disjoint_splits(**datasets)
    selected_rows = records_of(datasets["train"])
    sampler = ComponentBatchSampler([lookup[row["image_id"]] for row in selected_rows], 4, 42)
    schedule_lengths = []
    for epoch in range(4):
        sampler.set_epoch(epoch)
        assert sorted(i for batch in sampler for i in batch) == list(range(len(selected_rows)))
        schedule_lengths.append(len(sampler))
    protocol = {"status": "data_and_sampler_verified_NOT_execution_release", "purpose": "bounded target-domain data expansion, not architecture/loss hyperparameter search",
        "arms": ["fresh adapters", "ROCO adapters"], "same": ["architecture B", "loss weights", "original image + raw question input", "epoch count4", "seed42", "CPU8 float32 offline", "new task heads initialized identically within pair"],
        "TRAIN_components": 256, "TRAIN_stratum_components": {"positive": 128, "negative": 128},
        "selected_counts": {role: len(ds) for role, ds in datasets.items()}, "actual_manifests": {role: manifest(ds) for role, ds in datasets.items()},
        "four_epoch_sampler_no_drop_or_repeat": True, "micro_batches_per_epoch": schedule_lengths,
        "maximum_optimizer_steps_per_arm": 4 * ((schedule_lengths[0] + 3) // 4),
        "previous_TRAIN_subset_is_nested": True, "model_validation_QA_exactly_unchanged": True,
        "known_limitations": ["one seed and development-informed fold", "expanded TRAIN answer-vocabulary coverage may change head dimension; not a pure module causal comparison", "ROCO upstream compute not matched", "5positive pilot validation images remain uncertain; expanded321-image frozen fracture evaluation secondary", "no independent test or clinical evidence"],
        "required_before_launch": ["verify fresh/ROCO checkpoint lineage and byte-identical new task heads", "freeze selected TRAIN-derived vocabulary and task loss balance", "per-run config hashes and budget equality", "hash verify190-artifact technical reference", "confirm no running job", "retain frequency and always-no/yes controls"],
        "forbidden": ["PMC or memorization checkpoint initialization", "use calibration for model selection", "tune threshold on model-validation", "automatic long VLP training"],
        "calibration_reserved": True, "test_read": False, "training_released": False}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({key: value for key, value in protocol.items() if key != "actual_manifests"}, indent=2))


if __name__ == "__main__":
    main()
