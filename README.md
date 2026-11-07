# BoneVQA V4: detector-conditioned visual prompting for bone radiograph VQA

Reproducibility package for the manuscript *Detector-Conditioned Visual Prompting for Bone Radiograph Visual Question Answering: A Leakage-Audited Five-Fold Study With Mechanism Controls* (submitted to IEEE Access).

The package contains the complete source code, the component-grouped five-fold splits and question-answer files, the locked analysis protocols, and the stored per-image predictions of every arm. Every statistic reported in the paper is recomputed from these predictions by the analysis scripts. Model checkpoints and raw images are not included because of their size (see [Checkpoints and images](#checkpoints-and-images)).

## Repository layout

The layout mirrors the original project root so that the scripts run unchanged.

| Path | Contents |
|---|---|
| `refactor_v2/src/` | Model code: baseline B, the V4 local-token module, detector wrapper, data pipeline |
| `refactor_v2/experiments/` | Training, evaluation and analysis entry points (`access_*` = paper analyses) |
| `refactor_v2/scripts/`, `refactor_v2/configs/` | Training/detector scripts and YAML configurations |
| `refactor_v2/app/` | FastAPI service and web UI used for the latency and API tests |
| `refactor_v2/tests/` | Unit and gate tests |
| `refactor_v2/reports/access_v1/` | Locked protocol (`PROTOCOL.md`) and the paper's result files |
| `refactor_v2/reports/local_noaux_refinement_v1/` | Seed-42 five-fold predictions of B and V4 (main study) |
| `refactor_v2/reports/v5_studies/` | Seed-43 retrained arms (B, V4, V4_flag, V4_grid, V4_placebo) and FAVP-/LaPA-/Lens-style arms |
| `refactor_v2/reports/v4_detector_ext_v1/` | Post hoc detector-extension analysis (Section V-D) |
| `refactor_v2/reports/thesis_v4_checks_v1/`, `route_audit_v1/`, `v4_poststack_v1/` | Route interventions, leakage probe, robustness checks |
| `.evaluation_runtime/data/fracatlas_component_cv_v3/` | Five outer folds: TRAIN / model-validation / calibration / test QA files and component manifest |
| `.evaluation_runtime/data/fracatlas_component_clean/` | Component-clean manifests used for detector training |
| `.evaluation_runtime/data/external_grazpedwri/` | GRAZPEDWRI-DX metadata and file list for the external cohort |
| `exp4_multiseed/`, `mandatory_experiments_20261011/` | Code and protocols of the earlier sealed studies (leakage audit, Section IV-B) |
| `tools/relocate.py` | Rewrites the original absolute root to your clone location |

## Setup

```bash
git clone <this repository> bonevqa-v4 && cd bonevqa-v4
python tools/relocate.py          # scripts keep the original absolute root; rewrite it once
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # open_clip_torch 2.26.1 loads BiomedCLIP from the Hugging Face hub
```

The scripts were run with Python 3.11 on macOS (Apple M4 Pro). Training used the MPS backend; all evaluation and analysis ran on CPU.

## Reproduce the reported statistics (CPU, no GPU or images needed)

Run from `refactor_v2/`. Each command reads the stored predictions and rewrites the corresponding result file; `git diff` should show no numeric change.

| Paper content | Command | Output |
|---|---|---|
| Main results, folds, hardware-free subset, question families (Sec. IV-C) | `python -m experiments.access_stats_v1` | `reports/access_v1/stats.json` |
| Seed-43 replication and retrained ablation with Holm adjustment (Sec. IV-C, IV-E) | `python -m experiments.access_contrasts_v1 contrasts` | `reports/access_v1/contrasts.json` |
| Comparison with FAVP-, LaPA-, Lens-style arms (Sec. IV-D) | `python -m experiments.access_contrasts_v1 methods` | `reports/access_v1/methods.json` |
| Question wording and semantic router (Sec. IV-F) | `python -m experiments.access_routing_v1 analyze` | `reports/access_v1/routing/results.json` |
| Paraphrase set | `python -m experiments.access_paraphrase_analyze_v1` | `reports/access_v1/paraphrase/results.json` |
| External validation on GRAZPEDWRI-DX (Sec. IV-G) | `python -m experiments.access_external_v1 analyze` | `reports/access_v1/external/results.json` |
| Post hoc detector extension (Section V-D) | `python -m experiments.v4_consistency_v4arm_v1` | `reports/v4_detector_ext_v1/v4arm_full.json` |
| Post hoc availability strata and detector misses (Sec. IV-E) | `python -m experiments.access_availability_v1` | `reports/access_v1/availability.json` |

The first seven commands were re-run on a fresh copy of this package before the first release and reproduced the stored files with zero numeric differences; the availability analysis was checked the same way when it was added. Bootstrap analyses use fixed seeds recorded in the scripts.

## Retrain from scratch

Retraining requires the images and the initial checkpoints:

1. Download FracAtlas (Abedeen et al., 2023, CC BY 4.0) and GRAZPEDWRI-DX (Nagy et al., 2022, CC BY 4.0) and place the images where the manifests in `.evaluation_runtime/data/` point after relocation.
2. Train the per-fold detector on TRAIN only: `scripts/train_detector_v2.py --split_manifest ...` (see `refactor_v2/README.md`).
3. Train B and V4 with the configurations in `refactor_v2/configs/` (two epochs, AdamW 1e-4, batch 4 × 4 accumulation, seeds 42 and 43, last checkpoint).

The ROCO-aligned initialization and the trained fold checkpoints (about 250 MB each) are available from the authors on request and will be archived at [archive DOI, to be added].

## Checkpoints and images

Not included: raw radiographs (public datasets above), model checkpoints (`*.pt`), detector weights, SAM/MedSAM caches and the native assets of the compared methods (FAVP, LaPA, Localization Lens), which belong to their respective authors. Checkpoint SHA-256 hashes are recorded in the result folders.

## Data and licenses

- Code: MIT License (`LICENSE`).
- Derived split and QA files, predictions and result files: CC BY 4.0. They are derived from FracAtlas and GRAZPEDWRI-DX, both released under CC BY 4.0; please cite the original datasets.
- No patient-identifying information is contained in this repository; image identifiers are those of the public datasets.

## Citation

If you use this package, please cite the article (citation to be added on publication) and the M.S. thesis of the first author.
