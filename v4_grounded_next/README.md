# v4_grounded_next — True lesion grounding for BoneVQA V4

Separate workspace. `refactor_v2/` is **read-only**: it is imported, never written.
`vg/paths.guard_write` refuses any write outside this folder, and `env.sh` sends
bytecode and temp files here. A stat snapshot of all 6,533 files in `refactor_v2`
is compared before and after each session.

```bash
source v4_grounded_next/env.sh                       # always first
$PY tests/gradient_and_leakage_gate.py [--mps]       # 47 checks, about 30 s
$PY -m scripts.cross_fit_proposals smoke             # pipeline check, about 1 min
$PY -m scripts.cross_fit_proposals all --fold 0      # 5 inner detectors, about 1.7 h of CPU
```

## Audit of the sealed V4 code (read-only)

| Finding | Where | Consequence here |
|---|---|---|
| `FractureDetectorRuntime.predict` drops scores | `src/detector_runtime.py:475` | `vg/detector.predict_scored` copies its preprocessing and keeps the score, which s_max needs. On fold 0 it reproduces sealed `boxes.json` exactly (max \|Δ\| = 0.0). |
| V4 TRAIN proposals come from the detector trained on those same images | `local_noaux_refinement_v1.proposals` | Fixed by cross-fitting (below). |
| `LocalTokens` gate starts at σ(−4) ≈ 0.018 | `vnext_model_v1.py:16` | Crop tokens barely move the output at init: shifting the crop changes p by only 8e-5. The weak local signal is built into the starting point. |
| `adaptive_avg_pool2d(14→2)` | `vnext_model_v1.py:66` | Replaced by `pool_patches` (a reshape-mean that gives the same result on any device). |
| `import scripts.train_detector_v2` runs `setup_env_dirs("D:/BaoYen_work")`, which creates relative `D:/…` folders in the current directory | `src/utils.py:57` | `env.sh` sets TMP/TEMP/HF_HOME first and `cd`s to the root, so nothing is created. |
| 21/3371 images have EXIF orientation 3/6/8, but FracAtlas COCO boxes and polygons use raw-file coordinates | data | `vg/geometry` converts GT to the canonical frame before any IoU, mask or pointing-game computation. |

## Layout

```
v4_grounded_next/
  env.sh                      interpreter, offline HF, PYTHONPATH, pycache prefix, cwd guard
  vg/paths.py                 sealed input paths (sha-verified), write guard, component lookup
  vg/geometry.py              EXIF→canonical→letterbox maps, rasterisation, grid targets, IoU, grid fallback boxes
  vg/detector.py              V4 detector recipe (unchanged), score-keeping runtime
  vg/hard_negatives.py        same-radiograph, size- and intensity-matched negatives with IoU = 0
  vg/data.py                  public_batch (inference keys) vs training_supervision (TRAIN-only GT)
  vg/model.py                 VNextGroundedModel, GroundingHead, LesionPooling, RecoveryGate
  vg/registry.py              V4 / V4-CF / V4-G / V4-GC / V4-GCR / V4-GCR-flag (+ V4-GCR-grid)
  scripts/cross_fit_proposals.py
  tests/gradient_and_leakage_gate.py
  runs/  cache/               all outputs
```

Planned for Phase 2: `scripts/train_arm.py` (V4 schedule and budget, seeds 42/43/44),
`scripts/evaluate.py` (four metric groups plus size tertiles, paired component bootstrap
with 2,000 resamples), `scripts/occlusion_test.py` (three fill modes), `vg/backbones/`
(detector and encoder plug-ins).

## Inference contract and tensor shapes

At inference the model gets only the image, the question, and proposals from a detector
run on that image. With B = batch, d = 768, K ≤ 4 slots, T = 4, Hp = Wp = 14:

```
G  (B,197,d) ─┬─ Fusion(G,Q).x_f ─ GlobalFractureHead ─ p_global          (dashed bypass, never sees proposals)
Q  (B,L,d)    │
C  (B,K,4,d) ─┴─ G1 = G + σ(gate)·XAttn(G+W_q q, C)                        (V4 LocalTokens, unchanged)
                 ell = f_ground(G1_patches ⊙ (1+tanh W q) + XAttn(·, C content))  M̂ = σ(ell)
                       (no box map, no detector score: vg_ground_input='vlq')
                 z_les = Σ_p M̂_p G1_p / Σ_p M̂_p                           (G1 ⊙ M̂ spatial pooling)
                 G2 = [G1_cls + t_les(z_les), G1_p + M̂_p·U(G1_p)]          (t_les, U zero-init)
                 x_f = Fusion(G2,Q).x_f ──► ClosedHead, TypeHead, LLM prefix, FractureHead
                 logit_local = FractureHead(x_f) + h_les(z_les)            (h_les zero-init; vg_pool_read='feature')
p_final = (1−α)·sg(p_global) + α·p_local,   α = g_α(mean G, s_max, n/K, unc(C), max M̂, is_fallback)
          α ≡ 0 when m(I) = 0 (global fallback) | grid crops replace proposals (grid fallback)
```

M̂ reaches x_f in two ways (the z_les residual on CLS and the per-patch emphasis) and the
fracture logit in a third (h_les on the pooled vector z_les). M̂ is not a detached auxiliary head.

**Shortcut closure (2026-11-12).** The detector box map is not an input of the grounding
head: M̂ must come from vision–language features (G1, crop *content* C via cross-attention, Q).
No scalar summary of M̂ (max, mean, LSE) reaches the lesion token or the fracture logit;
the head sees M̂ only through pooled feature vectors. Both lesion routes and h_les are
zero-initialised, so V4-G and V4-GC start bit-identical to V4 (gate `P.identity_init_*`).
Legacy behaviour is kept behind `vg_ground_input='footprint'` and `vg_pool_read='scalar'`.
The recovery gate α still reads s_max, n/K, unc and max M̂ (it mixes p_global and p_local; it
is not the fracture head).
`L_ground` is computed after M̂ exists, from `gt_masks` that only `training_supervision`
builds.

**Flag control.** `vg_token_mode="flag"` replaces C with learned slot embeddings and
zeroes the box footprint. It keeps only m(I) and n(I), the strict availability/count
control. The gate shows its output is bit-identical under crop shift.

## Cross-fitting

For each outer fold, TRAIN is split with `StratifiedGroupKFold(5, seed=20261111)` on
(fractured, component_id); every component falls in exactly one inner fold (asserted).
Each inner detector uses the exact V4 recipe (MobileNetV3-320-FPN at 640, FrozenBN,
2 epochs, SGD 0.005, positive-anchored batch of 4, seed 42, CPU) and proposes only for
its held-out inner fold. model_val, calibration and test use the sealed V4 full-TRAIN
detector, re-run with scores and checked against sealed `boxes.json`. `audit.json`
compares TRAIN-CF, TRAIN-in-sample and TEST on m-rate (two-proportion z-test), best-IoU
distribution (KS test), recall at IoU 0.25/0.5, and s_max quantiles.

Cost is about 20 min per detector on CPU (V4 receipts: 1,040–1,330 s), so about 1.7 h
per outer fold and about 8.5 h for all 25 inner detectors.

## Hard negatives for L_contrast

R⁺ is the GT box with 15% context. R⁻ has the same w and h (±10%), sits on the same
radiograph, has zero intersection with every GT box, has mean grey > 40, at least 50%
bone foreground, and mean intensity within ±25% of R⁺. Sampling is deterministic per
(image_id, seed). If a constraint must be relaxed (intensity tolerance 0.6, then
foreground only), the relaxation is recorded for each negative. q_f is the projection
of `local_tokens.question(q)`, so L_contrast shapes the same question query the fusion
uses. τ = 0.07.

## MPS / numerical policy (M4 Pro, 24 GB)

* fp32 only. A construction-time assertion rejects any fp16 or bf16 parameter. There is
  no autocast; `llm_dtype=float32`, as in V4.
* No `adaptive_avg_pool2d`. Upsampling is bilinear with `align_corners=False`. GT grids
  are built in numpy on CPU. Reshaped tensors are made `.contiguous()` before conv.
* `PYTORCH_ENABLE_MPS_FALLBACK=0`, so a missing MPS kernel raises an error instead of
  silently running on CPU.
* MHA rows with all keys padded are avoided, using V4's `safe` mask.
* Detectors train on CPU, as in V4. VQA training may use MPS only after
  `gate --mps` passes; it currently does on torch 2.2.2: forward Δ = 6e-8, ground-head
  grad rel. error = 2e-7.
* All reported evaluation runs on CPU with `use_deterministic_algorithms`, 8 fixed
  threads and fixed seeds, followed by a replay check (V4 convention).

## Gate status (`runs/gates/gradient_and_leakage_gate.json`)

All 55 checks pass (P parity incl. identity init of G/GC, A gradients, S shortcut closure,
B causality, F flag control, C leakage, M CPU↔MPS). Eval-path parity: the sealed trained V4
checkpoint scored by `vg.trainer.evaluate_canonical` reproduces sealed model_val AUPRC
0.80103 exactly (max |Δp| 1.1e-5 on MPS).

**λ₂ scale (`runs/lambda/probe_fold0_mps.json`).** Against the *task* gradient, L_contrast at
λ₂ = 1 is 0.9× on the visual LoRA (cosine ≈ 0); the earlier "10³×" figure compared it with
L_ground. L_ground at λ₁ = 1 is 0.002× task on the LoRA. Sweep plan: `runs/lambda/sweep_plan_v2.json`.

## Scripts (Phase 2)

```bash
$PY -m scripts.train_arm --arm V4-GC --fold 0 --device mps --max-updates 719 --tag fold0/V4-GC_lc0.33 \
    --overrides '{"vg_lambda_contrast": 0.33}'
$PY -m scripts.select_lambda                    # pre-registered rule -> runs/lambda/selection.json
$PY -m scripts.mechanism_tests --fold 0         # T1 M̂ vs box map, T2 spatial permutation, T3 occlusion
```

## Trust layer (calibrated confidence and deferral), 2026-11-17

Adds nothing to the model and no forward pass for the yes / no endpoints (fracture, tumor, malignancy):

    source v4_grounded_next/env.sh; cd v4_grounded_next
    $PY -m scripts.trust_fit J_v4cf_yolojoint_btxrd          # fit on calibration / model_val scores, evaluate on test / external
    $PY -m scripts.answer_trust --name J_v4cf_yolojoint_btxrd --dataset btxrd --role test --image-id BT_IMG000470 --shortlist

* `vg/trust.py`: P(answer is correct) from the head margin (`head`) or the head margin plus the detector's top box score
  (`head_detector`, default); deferral below the validation quantile for 90 % or 80 % coverage; optional top-3 diagnosis list.
* `scripts/trust_fit.py` needs `<name>_{calibration,test}.json` (scripts.eval_arm) and `ref_<name>_<role>.json`
  (scripts.reference_score); writes `runs/joint/fold_0/trust_<name>.json`, `trust_<name>_eval.{json,md}`.
* `scripts/answer_trust.py --check` compares live probabilities with the stored evaluation rows (difference 0.0 on J).
* Limits: dataset images only (proposals are read from the proposal files); other question types are answered with a raw,
  uncalibrated score; thresholds come from about 300 validation images per endpoint.
