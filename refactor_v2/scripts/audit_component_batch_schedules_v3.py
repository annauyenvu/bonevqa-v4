"""Check common LF-control schedules on sealed TRAIN only; no model training."""
import hashlib
import json
import time
from pathlib import Path
from src.context_controls import component_balanced_batches, component_disjoint_donors
from src.data.sealed_manifest import load_sealed_records


def main():
    start = time.perf_counter()
    root = Path.cwd()
    seal = root / "refactor_v2/reports/STEP1_DATA_SEAL_V3_RUNTIME_VOCAB.json"
    component_path = root / ".evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json"
    expected = json.loads(seal.read_text())["artifacts"][component_path.relative_to(root).as_posix()]
    if hashlib.sha256(component_path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Component manifest hash mismatch")
    components = json.loads(component_path.read_text())["components"]
    lookup = {image: group for group, images in enumerate(components) for image in images}
    audits = []
    for fold in range(5):
        source = component_path.parent / ("fold_" + str(fold)) / "vqa_train.json"
        rows = load_sealed_records(source, seal, root)
        groups = [lookup[row["image_id"]] for row in rows]
        for seed in (42, 43):
            for batch_size in (4, 8):
                batches = component_balanced_batches(groups, batch_size, seed)
                donor_violations = 0
                for batch in batches:
                    identities = [groups[i] for i in batch]
                    donors = component_disjoint_donors(identities)
                    donor_violations += sum(identities[i] == identities[j] for i, j in enumerate(donors))
                audits.append({"fold": fold, "seed": seed, "batch_size": batch_size,
                    "qa": len(rows), "batches": len(batches), "cross_component_violations": donor_violations,
                    "schedule_sha256": hashlib.sha256(json.dumps(batches).encode()).hexdigest(),
                    "manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "last_batch_sizes": list(map(len, batches[-2:]))})
    result = {"scope": "TRAIN scheduling mechanics only; no gradient, efficacy or full training evidence",
              "component_manifest_sha256": expected, "audits": audits,
              "passed": all(row["cross_component_violations"] == 0 for row in audits),
              "seconds": time.perf_counter() - start,
              "training_entrypoint_enabled": False,
              "remaining": "wire common sampler and component IDs through every matched arm; checkpoint/resume and real-data route audit"}
    output = root / "refactor_v2/reports/component_batch_schedules_v3.json"
    if output.exists():
        raise RuntimeError("Existing schedule report must not be overwritten")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
