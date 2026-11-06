"""Derive detector roles from sealed VQA folds; annotations remain supervision."""
import json
from pathlib import Path

from src.data.sealed_manifest import load_sealed_records
from src.detector_protocol import file_sha256, load_detector_manifest


def main():
    root = Path.cwd()
    base = root / ".evaluation_runtime/data/fracatlas_component_cv_v3"
    seal = root / "refactor_v2/reports/STEP1_DATA_SEAL_V3_ALIGNMENT.json"
    metadata_path = base / "manifest.json"
    metadata = json.loads(metadata_path.read_text())
    components = {image: str(index) for index, group in enumerate(metadata["components"]) for image in group}
    reports = []
    for fold in range(5):
        directory = base / f"fold_{fold}"
        sources, split_ids, rows_by_id = {}, {}, {}
        for role, vqa_role in (("train", "train"), ("val", "model_val"),
                              ("calibration", "calibration"), ("test", "test")):
            path = directory / f"vqa_{vqa_role}.json"
            rows = load_sealed_records(path, seal, root)
            sources[role] = {"path": str(path), "sha256": file_sha256(path)}
            split_ids[role] = sorted({row["image_id"] for row in rows})
            rows_by_id.update({row["image_id"]: row for row in rows})
        payload = {"protocol_version": 3, "fold": fold, "scope": "development_informed_component_fold",
                   "project_root": str(root), "split_images": split_ids, "source_vqa": sources,
                   "source_seal": {"path": str(seal), "sha256": file_sha256(seal)},
                   "component_metadata_sha256": file_sha256(metadata_path),
                   "component_by_image": {image: components[image] for ids in split_ids.values() for image in ids},
                   "limitations": ["Development-informed evaluation, not independent clinical validation",
                                    "Train detector only on train; select only on val; calibration never selects weights"]}
        output = directory / "detector_split_v3.json"
        if output.exists():
            raise RuntimeError("Detector manifest already exists; inspect rather than overwrite: " + str(output))
        output.write_text(json.dumps(payload, indent=2) + "\n")
        load_detector_manifest(output)
        reports.append({"fold": fold, "path": str(output), "sha256": file_sha256(output),
                        "images_by_role": {role: len(ids) for role, ids in split_ids.items()}})
        if fold == 0:
            # All 12 smoke images come from the outer TRAIN partition, and
            # singleton components ensure no hidden component overlap.
            singleton = {group[0] for group in metadata["components"] if len(group) == 1}
            positive = [image for image in split_ids["train"] if image in singleton and rows_by_id[image]["fractured"]]
            negative = [image for image in split_ids["train"] if image in singleton and not rows_by_id[image]["fractured"]]
            smoke = dict(payload)
            smoke.update(scope="technical_smoke_train_only",
                         split_images={"train": positive[:2] + negative[:6],
                                       "val": positive[2:3] + negative[6:7],
                                       "test": positive[3:4] + negative[7:8]},
                         source_vqa={role: sources["train"] for role in ("train", "val", "test")})
            smoke_path = directory / "detector_smoke_train_only_v3.json"
            smoke_path.write_text(json.dumps(smoke, indent=2) + "\n")
            load_detector_manifest(smoke_path)
    report = {"status": "prepared_and_partition_verified", "folds": reports,
              "smoke_scope": "12 TRAIN-only images in disjoint singleton components; not evaluation evidence"}
    Path("refactor_v2/reports/detector_fold_bindings_v3.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
