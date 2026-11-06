# BoneVQA-Prompt v5 — fidelity map to LaPA, FAVP and Localization Lens

## Experimental design (decided 2026-10-30)

v5 ablates prompt TYPES, not each paper's internals: a 2^4 factorial over V (FAVP), L (Lens),
P (LaPA) and T (textual prompt), every factor in the best configuration its paper reports,
plus the sealed v4 local route as a reference. Per-paper internal ablations (prompt kinds,
mask sources, pixel-shuffle factor, phase-1 loss, graph controls, fusion mode) are NOT run.

| Factor | Configuration used | Source in paper |
|---|---|---|
| V | SAM-Med2D AVPC, 50 keypoints, NMS 0.80, repo code-default filters 0.68/0.70 (author `train_rad.yaml`: pred_iou 0.8, stability 0.9), top-3 instances by predicted IoU; circle prompt; HAG (local + global tokens -> Q-Former -> LLM); ViT LoRA r4, LLM LoRA r8 | FAVP Fig. 3-4, Tables 3-4 |
| L | MedSAM masks only (paper: SAM-ViT-H + MedSAM; see scope note below); unicolor / multicolor / masked views, 2 random + original; DCL alignment to region name, then QA InfoNCE; pixel shuffle r = 4; original-image inference | Lens Sec. 2, Table 3 best row |
| P | GM + LF (L -> I -> MM) + PF; latent 32; eta 0.1; alpha 1, theta = beta = 0.1; PF graph = bone graph from TRAIN | LaPA Sec. 3, Tables 2-5, Fig. 3 |
| T | fixed clinical instruction without region / label words | project template (no answer leakage) |

Domain adaptations that remain necessary: bone graph instead of SLAKE (P); FracAtlas region names as
the organ text for Lens alignment (L); MedSAM-only Lens masks (L); top-3 AVPC instances to bound compute (V). Analysis: each cell
vs B (paired component bootstrap, model-validation for screening), main effects and two-way
interactions over the 16 cells. Rows below that mention per-paper ablation arms describe code that
exists but is not part of the planned study.


## Scope reduction for the manuscript (decided 2026-10-31)

The remaining 12 cells needed 47-62 h on this Mac; the manuscript is due 2026-11-01. Run only V, L and VLPT
next to the finished B, T, P, PT, V4 (fold 0, seed 42, exploratory). Main effects, two-way interactions and the
`confirm` suite are NOT available. L uses MedSAM masks only, because SAM-H was stopped at 753/4013 images.
The primary evidence stays the sealed 5-fold V4 vs B study (`reports/local_noaux_refinement_v1`).

AVPC top-3 quality on fold-0 TRAIN (2568 images, 7700 boxes, "dark" = below Otsu, may include soft tissue):
47.4% of boxes dark on > 90% of their area, 66.8% on > 50%; 45.1% of images have duplicate boxes (IoU > 0.5);
458 lesion images: any box IoU >= 0.1 3.7% (other-image control 3.3%), >= 0.3 1.1% (0.7%), lesion centre inside
a box 10.7% (18.4%). On 100 images, all regions (median 22) cost ~6x the views with no lesion gain over control.

Date: 2026-10-30. Code: `refactor_v2/experiments/v5/`. Tests: `refactor_v2/tests/test_v5_model.py`.
Sealed sources (`src/`, v4 study, earlier pilots) are not modified; v5 subclasses `BoneVQAModel`.

## Deviations found in earlier ports and how v5 fixes them

| # | Paper mechanism | Earlier BoneVQA implementation | v5 fix | Where | Verified by |
|---|---|---|---|---|---|
| 1 | FAVP HAG: local and global ViT tokens concatenated, (N+1)xP, into the Q-Former that feeds the LLM (Eq. 5-7) | v4 fed local tokens only into the fusion via a gated residual; the Q-Former saw global tokens only, so local information reaches the LLM only through the fusion token x_f in the prefix | `v5_hag=replace`: Q-Former memory = global tokens + full patch tokens of every local view | `model_v5._hag_prefix` | `test_hag_feeds_local_tokens_into_qformer` |
| 2 | FAVP visual prompt kinds (Table 4: mask, circle, box, contour, ...) | v4 used crops only | `v5_local_view` in crop / box / circle / contour / mask | `model_v5._local_views` | arms `HAG_box`, `HAG_circle`, `HAG_contour`, `HAG_mask` |
| 3 | FAVP AVPC: SAM-Med2D grid keypoints, NMS tau tuned (Fig. 4: 50 keypoints, tau 0.80) | port used pred_iou 0.80 / stability 0.90 (repo code defaults are 0.68 / 0.70) and no tuning; zero masks on X-rays | author default filters; sweep keypoints {30,40,50,60} x tau {0.75..0.90} on TRAIN images only | `masks_v5 avpc_tune / avpc_generate` | arm `HAG_avpc`, `avpc_tune_fold0.json` |
| 4 | Lens views from anatomical segmentation (SAM + MedSAM), unicolor / multicolor / masked, 2 random views + original, alpha ~ N(0.5, 0.1) | views built from detector lesion boxes | anatomy masks for every image; primary source MedSAM (ViT-B, medical); SAM-ViT-H + MedSAM (paper setting, author default `facebook/sam-vit-huge`) is an ablation promoted only if it beats MedSAM-only; Otsu = non-learned control; optional lesion view with label-blind placebo for negatives | `masks_v5 anatomy`, `study_v5.lens_mask_loader`, `model_v5.lens_loss` | arms `HAG_LENS`, `HAG_LENS_samh_medsam`, `HAG_LENS_otsu`, `HAG_LENS_lesion` |
| 5 | Lens phase 1: DCL between views and the organ name | QA-text alignment only | phase 1 = one TRAIN pass of region-text DCL (`forward_lens_align`); phase 2 = QA InfoNCE | `model_v5.lens_loss`, `study_v5.run_job` | `HAG_LENS` vs `HAG_LENS_infonce` |
| 6 | Lens pixel shuffle factor 4 | factor 2 | r = 4 with zero-padding of the 14x14 grid to 16x16 (r = 2 kept as ablation) | `modules_v5.LensFusionV5` | `test_lens_fusion_shapes`, arm `HAG_LENS_r2` |
| 7 | Lens: inference on the original image only | already training-only | kept; enforced by test | `model_v5` | `test_lens_never_changes_inference` |
| 8 | LaPA PF: GNN over an organ-disease graph relevant to the data | SLAKE 409-node graph (chest / abdomen / brain) | bone graph: 49 nodes (regions, bones, joints, views, findings, counts, locations), anatomy + TRAIN-only co-occurrence edges; GAT with neighbour masking; `CA(F_G, X_LP)` as in Eq. 11 | `bone_kg_v5`, `modules_v5.BoneKnowledgePrior` | `test_prior_uses_graph_structure`, arm `HAG_GMLF_PF` vs `_rewired` / `_empty` |
| 9 | LaPA Eq. 12: X_F = alpha X_II + theta F_FI + beta F_FL, alpha 1, theta = beta = 0.1 | latent as a zero-initialised gated residual on the pooled baseline | `v5_latent_mode=lapa` implements Eq. 12; `residual` kept as ablation | `model_v5._lapa` | arm `HAG_GMLF_PF_residual` |
| 10 | LaPA consistency weight eta = 0.1 (Table 5), latent size 32 (Table 4) | eta = 0.01 | eta = 0.1, n_latent = 32; eta = 0 control | `V5_DEFAULTS` | arm `HAG_GMLF_nocs` |
| 11 | Deployment of a local-route model | `app/inference.py` loaded every checkpoint as `BoneVQAModel` and loaded the detector only when `use_visual_prompt`; a v4 checkpoint would silently run without crops (= B) | engine loads the checkpoint's own class and fails closed without the detector | `inference_v5.py`, `app/inference.py` | `test_engine_detects_local_route_for_v4_and_v5` |

Note on v3/v4: both sealed studies ran with `sam.enabled = false`; their ROIs were detector
rectangles (MedSAM was configured but never executed). SAM-ViT-H appeared only in the 28/8-image
native Lens pilot. v5 introduces segmentation for Lens views as a training-only input.

## Compatibility guarantees

* All `v5_*` switches off reproduces B bit-for-bit (`test_all_off_equals_B`).
* Only `v5_local` reproduces v4 bit-for-bit (`test_local_only_equals_v4`) and the sealed fold-0
  v4 / B checkpoints give identical predictions inside `BoneVQAv5` (`check_real_parity_v5.py`).

## Device policy

torch 2.6 MPS qualification (accel_v1, 2026-10-30): training loss / gradients match CPU
(|dloss| 3e-6, fracture probability 2.4e-7, repeat-exact), but greedy OPEN generation on MPS is
corrupted. v5 therefore trains on MPS and runs every evaluation, generation and replay on CPU.
All arms of a suite share one device; MPS-trained arms are never compared with CPU-trained arms.

**Correction (2026-10-30 evening).** The accel_v1 check compared loss and fracture probability only;
loss agrees to ~1e-4 even with the two torch 2.6 MPS bugs below, so it could not detect them.
Every MPS training result before this date (factorial_fold0 B, T, P; failed PT, V4) was deleted.

| MPS bug (torch 2.6) | Effect | Fix | Verified by |
|---|---|---|---|
| `layer_norm` backward with an input that does not require grad returns wrong weight/bias grads (0, garbage, inf) | `fusion.txt_sa.norm1` (all arms), `v5_gm.answer_sa.norm1` (P), `local_tokens.norm` (V4/V); NaN in PT at update 43 and V4 | `model_v5._mps_safe_layer_norm` gives such inputs a discarded grad on MPS only | `test_v5_mps.py`, `mps_grad_qualification_{fix,nofix}.json` |
| `adaptive_avg_pool2d` forward on the strided patch-grid view, \|err\| ~0.6 | wrong local crop features while training V4 / V cells | `.contiguous()` before pooling, MPS only (CPU stays bit-identical to v4) | `test_v5_mps.py` |

Every MPS job now starts with a hard per-parameter qualification inside `study_v5.gate`
(`qualify_mps_grads_v5.qualify`): same weights and first 4 TRAIN micro-batches on CPU and MPS,
dropout off; each trainable parameter with gradient norm >= 1e-6 must satisfy
||g_mps - g_cpu|| / ||g_cpu|| <= 1e-2 (observed after the fixes: <= 3e-3 over B, P, V4, V, L, VLPT;
CPU 8-thread vs 1-thread differs by <= 3e-6). Without the fixes the gate fails as intended.
CPU assessment and fresh-process replay now use the same thread count (recorded in `trained.json`);
at 8 vs 4 threads PT differed by 6e-8 on 17/400 rows, answers equal, and by 0 at equal thread counts.
MedSAM masks generated on MPS: 6/10 sampled images bit-identical to CPU, the rest differ only on
boundary pixels (best-match IoU >= 0.9995); AVPC boxes identical on the 3 checked images.

## Inference input contract (final solution)

Radiograph (EXIF applied) + English question from the trained families. Detector proposals are
computed internally. Annotations, lens views and answers are never inputs.
