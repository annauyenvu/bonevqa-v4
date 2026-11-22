# BoneVQA: detector-conditioned visual prompting for bone radiograph VQA (V4 and final system JF)

Reproducibility package for the manuscript *Detector-Conditioned Visual Prompting for Bone Radiograph VQA: A Leakage-Controlled Evaluation and Mechanism Analysis* (in preparation for submission to IEEE Access).

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
| `tools/repro_check.sh`, `tools/compare_json.py` | Clean-clone reproduction test of the analysis commands |

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
| Post hoc proposal count, late fusion with component-bootstrap CIs (Sec. IV-E) and exact-match accuracy CI (Sec. IV-C) | `python -m experiments.access_review2_v1` | `reports/access_v1/review2.json` |
| Third-revision checks: proposals by fold role (Sec. IV-E), mean of folds and within-fold ranks (Sec. III-I, IV-C), paired component permutation tests (Sec. IV-C, IV-E), negation exact match and sensitivity by detector score (Supplementary S1, S2) | `python experiments/access_review3_v1.py .. reports/access_v1/review3/posthoc.json reports/access_v1/review3/detector_scores.json` | `reports/access_v1/review3/posthoc.json` |

The first seven commands were re-run on a fresh copy of this package before the first release and reproduced the stored files with zero numeric differences; the post hoc availability and late-fusion analyses were checked the same way when they were added. Bootstrap and permutation analyses use fixed seeds recorded in the scripts.

`tools/repro_check.sh <this repository> <empty work dir> <python>` repeats that clean-clone test for every command above: it exports the committed tree to a new path, runs `tools/relocate.py`, checks that the inputs exist, reruns each command and compares the regenerated file with the committed one (`tools/compare_json.py`; path strings are ignored).

Two third-revision inputs need data that are not in this package, so their outputs are stored instead:

| Paper content | Script | Needs | Stored output |
|---|---|---|---|
| Detector scores of the kept proposals (input of the last row above; Supplementary S1) | `experiments/access_detector_rescore_v1.py` | fold detector checkpoints, FracAtlas images | `reports/access_v1/review3/detector_scores.json` |
| Overlap of FracAtlas with the ROCO and PMC-VQA pretraining pools (Sec. III-C, Supplementary Fig. S1) | `experiments/access_pretraining_overlap_v1.py` | FracAtlas, ROCO and PMC-VQA images | `reports/access_v1/review3/pretraining_overlap.json` |

## Final system (JF): YOLOv8, bone tumors and confidence layer (added November 2026)

This part of the package backs the final configuration of the M.S. thesis and of the companion manuscript *Localization Regime, Not Box Accuracy, Drives Detector-Conditioned Prompting in Bone Radiograph Visual Question Answering* (in preparation). JF is the V4 local-token model with a two-class YOLOv8s detector (fracture, bone lesion), cross-fitted training proposals for FracAtlas (BTXRD training proposals are in-sample), the detector's own proposal levels (fracture r = 0.02 in training and inference; lesion r = 0.02 in training and r = 0.10 at inference), joint training on FracAtlas-VQA and BTXRD-VQA, and a confidence layer that uses the highest detector score.

| Path | Contents |
|---|---|
| `v4_grounded_next/vg/` | Model wrapper with leakage guard, joint FracAtlas/BTXRD stream (`joint.py`), knowledge latent prompt (`kg.py`), confidence layer (`trust.py`) |
| `v4_grounded_next/scripts/` | Training (`train_arm.py`, `train_joint.py`), evaluation (`eval_arm.py`, `eval_joint.py`), comparisons (`compare_arms.py`, `compare_btxrd.py`), cross-fitting, confidence layer, reference-information and knowledge analyses |
| `v4_grounded_next/runs/arms/` | Training records of every arm and fold (`started.json`, `completed.json` with checkpoint SHA-256) |
| `v4_grounded_next/runs/yolo_ablation/fold_<f>/` | Per-image fracture predictions of every arm on test and calibration, and paired-bootstrap comparisons |
| `v4_grounded_next/runs/joint/fold_<f>/` | Per-image BTXRD predictions (test, external, model-val), comparisons, confidence-layer fits and evaluations, knowledge and reference analyses |
| `v4_grounded_next/runs/crossfit*/` | Proposal files: Faster R-CNN and YOLOv8 cross-fitted proposals, level sweep (`crossfit_yolo_joint_sweep/r<level>/`) |
| `v4_grounded_next/demo_jf/` | Deployed service (FastAPI + YOLOv8 worker), API tests, latency benchmark, parity check |
| `yolo_pilot/code/` | Detector data preparation, training and prediction, BTXRD-VQA generator, proposal builders, run lanes, summary tables |
| `yolo_pilot/datasets/` | Fold image lists, inner cross-fitting lists, box annotations in JSON (`gt.json`), BTXRD-VQA question-answer files and vocabulary (images and label files not included) |
| `yolo_pilot/runs/pred/`, `eval/`, `frcnn/` | Raw detector predictions of every fold and detector evaluations |
| `yolo_pilot/runs/summary_final/` | `summary.json` and `bang_tong_hop.md`: every five-fold number of the final system |
| `paper_jf/` | Post hoc analyses of the manuscript: detector × training-regime factorial and same-image comparison of all methods |
| `thesis_jf/` | Script that turns `summary.json` into the number cells of the thesis (`results_jf.json`) |

### Offline procedure for one outer fold (Algorithm 1 of the manuscript)

The steps below are what `yolo_pilot/code/lane_gpu.sh`, `lane_cpu.sh` and `lane_final.sh` run; step numbers follow Algorithm 1.

1. Inputs: the FracAtlas-VQA roles of fold f (TRAIN, model-val, calibration, test), BTXRD-VQA (train, val, test, external) and the lesion boxes of training images. The proposal levels were fixed once on fold 0 (step 5).
2. Train the main detector D_f on FracAtlas TRAIN and all BTXRD train images (`yolo_pilot/runs/yolo/fa_btxrd_os3[_f<f>]`, weights not included).
3. Cross-fitting, FracAtlas only: split FracAtlas TRAIN into three component-grouped inner folds (`yolo_pilot/datasets/fa_fold<f>/cf/`). Each inner detector is trained on the other two inner folds **plus all BTXRD train images**, and predicts proposals only for the FracAtlas images of its held-out inner fold.
4. With D_f, predict proposals for FracAtlas model-val, calibration and test, and for every BTXRD image (val, test, external and train). The BTXRD train proposals are therefore in-sample (`crossfit_yolo_joint_sweep/r<level>/btxrd_fold<f>/proposals_btxrd_insample.json`).
5. Display threshold per class: maximum box F1 at IoU 0.5 on model-val (`summary.json`, `detector`). Proposal levels: fracture r = 0.02 in training and inference (chosen on fold 0 model-val, `yolo_pilot/runs/sweep_analysis/`); lesion r = 0.02 in training and r = 0.10 at inference (chosen on fold 0 BTXRD val, `v4_grounded_next/runs/joint/final_btxrd_level.json`).
6. Train the answering model on FracAtlas-VQA and BTXRD-VQA (two epochs, last checkpoint): FracAtlas TRAIN images use the cross-fitted proposals of step 3 (`crossfit_yolo_joint_sweep/r0.02/fold_<f>/proposals_fold<f>.json`); BTXRD train images use the in-sample proposals of D_f from step 4 (`r0.02/btxrd_fold<f>/proposals_btxrd_insample.json`).
7. Fit the confidence layer on FracAtlas calibration and BTXRD val with the step-4 proposals at the inference levels (`v4_grounded_next/runs/joint/fold_<f>/trust_JF_*.json`).

### Where each reported result comes from

Run from the repository root after `python tools/relocate.py`. The commands marked *CPU* read only stored files; they were rerun on a clean copy of this repository and reproduced the committed files with zero numeric difference.

| Result (thesis table label; manuscript table) | Stored files | Command |
|---|---|---|
| Detector comparison on fold 0 (`detcmp`; Table VIII) | `yolo_pilot/runs/eval/fold0_fa.json`, `fold0_cross.json` | `yolo_pilot/code/eval_detectors.py`, `eval_cross.py` (need images and weights) |
| Detector at its own operating point and at the proposal levels, five folds (`detop`; Table IX) | `yolo_pilot/runs/summary_final/summary.json` (`detector`) | *CPU* `python3 -I yolo_pilot/code/summary_final.py` |
| Fold-0 decomposition and detector × regime factorial (`decomp`; Table X) | `v4_grounded_next/runs/yolo_ablation/fold_0/*_test.json`; `paper_jf/results/regime_factorial_fold0.json` | *CPU* `python -I paper_jf/analysis/regime_factorial_fold0.py` |
| Proposal levels with placebo and oracle boxes (`jfground`; Table XI) | `yolo_pilot/runs/sweep_analysis/fold0.json`; `yolo_ablation/fold_0/compare_test_loc_r*.json` | `yolo_pilot/code/sweep_analysis.py report` |
| Five-fold ladder V4 → JF, hardware-free subset (`jffolds`, `jfhw`; Table XII) | `yolo_pilot/runs/summary_final/summary.json`, `bang_tong_hop.md`; `yolo_ablation/fold_<f>/compare_test_*.json` | *CPU* `python3 -I yolo_pilot/code/summary_final.py` |
| Same-image comparison with image-only classifiers and FAVP/LaPA/Lens-style arms (Table XIV) | `paper_jf/results/compare_methods.json` | *CPU* `python -I paper_jf/analysis/compare_methods.py` |
| Bone-tumor questions and per-class recall (`jfbtxrd`, `jfclass`; Table XV) | `v4_grounded_next/runs/joint/fold_<f>/JF_*_btxrd_{test,external}.json`, `compare_btxrd_*_JF_vs_J.json` | *CPU* `summary_final.py` |
| Knowledge latent prompt and reference information (`kgres`, `refres`; Table XVI) | `joint/fold_0/kg_report_K.json`, `compare_btxrd_*_kg.json`, `ref_*.json` | `v4_grounded_next/scripts/kg_report.py`, `reference_analyze.py` |
| Confidence layer (`trustjf`; Table XVII) | `joint/fold_<f>/trust_JF_*_eval.json`, `trust_J_*_eval.md` | `v4_grounded_next/scripts/trust_fit.py`, `selective_report.py` |
| Deployed service: latency, API tests, parity (`latency`, `apitests`; Table XVIII) | `v4_grounded_next/demo_jf/benchmark.json`, `api_tests.json` | `benchmark.py`, `api_tests.py`, `parity_check.py` (need weights) |
| All number cells of the thesis results chapter | `thesis_jf/results_jf.json` | *CPU* `python3 -I thesis_jf/gen_jf.py` |

The V4-era results (main five-fold study, seed-43 controls, late fusion, availability strata, GRAZPEDWRI-DX, question wording) are reproduced by the commands in the previous section.

Not included for this part: images, detector weights (`best.pt`) and VQA checkpoints (SHA-256 in `runs/arms/**/completed.json`), and the bone knowledge graph and its 47 knowledge cards. The graph is mapped to the Radiology Gamuts Ontology, whose CC BY-NC-ND license does not allow redistribution of a derived graph; the analyses report only the graph and card-bank hashes.

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
- Derived split and QA files, predictions and result files: CC BY 4.0. They are derived from FracAtlas, GRAZPEDWRI-DX and BTXRD (Yao et al., 2025, figshare 10.6084/m9.figshare.27865398), all released under CC BY 4.0; please cite the original datasets.
- No patient-identifying information is contained in this repository; image identifiers are those of the public datasets.

## Citation

If you use this package, please cite the article (citation to be added on publication) and the M.S. thesis of the first author.
