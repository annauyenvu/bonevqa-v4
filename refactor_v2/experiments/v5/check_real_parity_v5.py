"""Real-weight parity: sealed v4/B fold-0 checkpoints reproduce exactly inside BoneVQAv5.

Loads the sealed LAST checkpoints with their original class and with BoneVQAv5
(same trainable tensors) and compares predictions on 16 model-validation rows.
Writes refactor_v2/reports/v5_studies/real_parity.json.  CPU, no training.
"""
import json
from pathlib import Path

import torch

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
V4 = ROOT / "refactor_v2/reports/local_noaux_refinement_v1"
OUT = ROOT / "refactor_v2/reports/v5_studies/real_parity.json"


def rows(model, batcher_ds, collate, spec, n=16):
    from experiments.v5.study_v5 import predict
    return predict(model, batcher_ds, collate, spec, "model_val", limit=n)


def main():
    from experiments.v5.model_v5 import BoneVQAv5
    from experiments.v5.study_v5 import ARMS, batcher
    from experiments.vnext_model_v1 import VNextModel
    from src.models.bonevqa import BoneVQAModel

    torch.set_num_threads(8)
    report = {}
    for arm, path, cls, override in (
        ("V4", V4 / "fold_0/local_real_noaux/LAST.pt", VNextModel, {"v5_local": True}),
        ("B", V4 / "fold_0/B/LAST.pt", BoneVQAModel, {}),
    ):
        spec = ARMS[arm]
        ds, collate = batcher("0", spec)
        ref_model = cls.load(str(path), device="cpu")
        ref = rows(ref_model, ds, collate, spec)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = {k: v for k, v in payload["config"].items() if not k.startswith("vnext_")}
        cfg.update(override)
        v5 = BoneVQAv5(cfg, payload["answer_vocab"], device="cpu")
        missing, unexpected = v5.load_state_dict(payload["state"], strict=False)
        assert not unexpected, unexpected
        got = rows(v5.eval(), ds, collate, spec)
        report[arm] = {"rows": len(ref), "identical": got == ref,
                       "max_p_yes_diff": max([abs(a.get("p_yes", 0) - b.get("p_yes", 0)) for a, b in zip(got, ref)] or [0])}
        del ref_model, v5
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert all(v["identical"] for v in report.values()), "v5 does not reproduce sealed checkpoints"


if __name__ == "__main__":
    main()
