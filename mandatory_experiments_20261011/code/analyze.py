#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from runtime import ROOT, configure


SEEDS = [42, 123, 456, 789, 2026]
VARIANTS = ["none", "v", "p", "vp"]
METRICS = ["overall", "closed", "open"]


def correct(engine, row):
    if row["answer_type"] == "CLOSED":
        return engine.normalize_yes_no(row["pred"]) == engine.normalize_yes_no(row["gt"])
    return engine.normalize_answer(row["pred"]) == engine.normalize_answer(row["gt"])


def hierarchical(nums, dens, reps=10000, seed=20261011):
    rng = np.random.default_rng(seed)
    nseed, nvariant, nmetric, nimage = nums.shape
    bootstrap = np.empty((reps, nvariant, nmetric), dtype=np.float64)
    for start in range(0, reps, 100):
        batch = min(100, reps - start)
        seed_draws = rng.integers(nseed, size=(batch, nseed))
        values = np.zeros((batch, nvariant, nmetric), dtype=np.float64)
        for slot in range(nseed):
            image_draws = rng.integers(nimage, size=(batch, nimage))
            for metric in range(nmetric):
                denominator = dens[metric, image_draws].sum(axis=1)
                for variant in range(nvariant):
                    selected = nums[seed_draws[:, slot], variant, metric, :]
                    values[:, variant, metric] += np.take_along_axis(selected, image_draws, axis=1).sum(axis=1) / denominator
        bootstrap[start:start + batch] = 100 * values / nseed
    return bootstrap


def main() -> None:
    common, engine, specs = configure()
    common.setup(42)
    common.verify_lock()
    output = ROOT / "analysis"
    output.mkdir(exist_ok=True)
    test = common.read(ROOT / "protocol/test_manifest.json")
    image_ids = sorted({row["image_id"] for row in test})
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    nums = np.zeros((len(SEEDS), len(VARIANTS), len(METRICS), len(image_ids)), dtype=np.float64)
    dens = np.zeros((len(METRICS), len(image_ids)), dtype=np.float64)
    for row in test:
        index = image_index[row["image_id"]]
        dens[0, index] += 1
        dens[1 if row["answer_type"] == "CLOSED" else 2, index] += 1

    per_seed = []
    for seed_index, seed in enumerate(SEEDS):
        for variant_index, variant in enumerate(VARIANTS):
            run = ROOT / "detector" / variant / f"seed_{seed}"
            qc = common.read(run / "QC.json")
            assert qc["valid"] and qc["component_clean"] and qc["seed"] == seed and qc["config"] == variant
            records = common.read(run / "test_predictions.json")
            engine.check_predictions(records, "test")
            for row in records:
                index = image_index[row["image_id"]]
                value = float(correct(engine, row))
                nums[seed_index, variant_index, 0, index] += value
                nums[seed_index, variant_index, 1 if row["answer_type"] == "CLOSED" else 2, index] += value
            metrics = common.read(run / "metrics.json")
            per_seed.append({"seed": seed, "variant": variant,
                             **{key: value for key, value in metrics.items() if isinstance(value, (int, float))}})

    points = 100 * nums.sum(axis=-1) / dens.sum(axis=-1)[None, None, :]
    boots = hierarchical(nums, dens)
    v_none_seed = points[:, VARIANTS.index("v"), :] - points[:, VARIANTS.index("none"), :]
    interaction_seed = (points[:, VARIANTS.index("vp"), :] - points[:, VARIANTS.index("p"), :]
                        - points[:, VARIANTS.index("v"), :] + points[:, VARIANTS.index("none"), :])
    v_none_boot = boots[:, VARIANTS.index("v"), :] - boots[:, VARIANTS.index("none"), :]
    interaction_boot = (boots[:, VARIANTS.index("vp"), :] - boots[:, VARIANTS.index("p"), :]
                        - boots[:, VARIANTS.index("v"), :] + boots[:, VARIANTS.index("none"), :])

    contrasts = []
    for metric_index, metric in enumerate(METRICS):
        for name, seeds, boot in (("V - None", v_none_seed[:, metric_index], v_none_boot[:, metric_index]),
                                  ("VxP DiD at L=0", interaction_seed[:, metric_index], interaction_boot[:, metric_index])):
            low, high = np.quantile(boot, [.025, .975])
            contrasts.append({"contrast": name, "metric": metric, "mean_difference_pp": float(seeds.mean()),
                              "seed_sd_pp": float(seeds.std(ddof=1)), "ci95_low_pp": float(low),
                              "ci95_high_pp": float(high), "positive_seeds": int((seeds > 0).sum()),
                              "negative_seeds": int((seeds < 0).sum()), "individual_seed_values": seeds.tolist()})

    cells = []
    for variant_index, variant in enumerate(VARIANTS):
        for metric_index, metric in enumerate(METRICS):
            values = points[:, variant_index, metric_index]
            cells.append({"variant": variant, "metric": metric, "mean_pct": float(values.mean()),
                          "seed_sd_pp": float(values.std(ddof=1)), "min_pct": float(values.min()),
                          "max_pct": float(values.max()), "individual_seed_values": values.tolist()})

    common.write(output / "results.json", {"protocol": {"bootstrap_reps": 10000, "bootstrap_seed": 20261011,
                                                          "unit": "paired training seed x image cluster"},
                                                    "cells": cells, "contrasts": contrasts, "per_seed": per_seed})
    np.savez_compressed(output / "bootstrap_replicates.npz", cell_accuracy_pct=boots,
                        v_minus_none_pp=v_none_boot, interaction_vxp_pp=interaction_boot)
    with (output / "contrasts.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["contrast", "metric", "mean_difference_pp", "seed_sd_pp", "ci95_low_pp", "ci95_high_pp",
                  "positive_seeds", "negative_seeds", "individual_seed_values"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(contrasts)

    lines = [
        "# Mandatory clean-split Detector factorial results",
        "",
        "Five paired training seeds; conservative exact+perceptual component split; 10,000 hierarchical seed x image-cluster bootstrap replicates.",
        "",
        "| Contrast | Metric | Mean difference (pp) | Seed SD | 95% CI | Positive seeds |",
        "|---|---|---:|---:|---|---:|",
    ]
    for row in contrasts:
        lines.append(f"| {row['contrast']} | {row['metric']} | {row['mean_difference_pp']:+.3f} | {row['seed_sd_pp']:.3f} | [{row['ci95_low_pp']:.3f}, {row['ci95_high_pp']:.3f}] | {row['positive_seeds']}/5 |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
