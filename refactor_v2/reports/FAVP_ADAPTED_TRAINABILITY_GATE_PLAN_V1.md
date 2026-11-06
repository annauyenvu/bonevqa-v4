# FAVP adapted native-route training gates — prospective next steps

The full CPU FP16 bridge and projector update/reload are accepted technical checks. Temporary FP32 frozen-linear compute resolves the measured CPU kernel bottleneck, but has small measured numerical differences and is not bitwise FP16 equivalence. No quality release exists.

## Source trainability versus proposed memory-safe implementation

| Component | Pinned author RAD route | Next technical gate policy |
|---|---|---|
| SAMMed2D | Frozen automatic proposals | Frozen, native unchanged thresholds and completed TRAIN census |
| EVA | `freeze_vit=True` attaches qkv LoRA r4/alpha8/dropout .05; vision norm frozen | Preserve LoRA intent where feasible; do not call base-only cached EVA a full source training route |
| QFormer/query tokens | RAD configuration trainable | Test actual trainability and gradients, not only shape |
| Vicuna | q_proj/v_proj LoRA r8/alpha16/dropout .05 | Frozen FP16 base with temporary FP32 base-linear compute; trainable FP32 low-rank branches must retain their own gradients |
| Image-to-LM projection | Loaded stage handoff if supplied | Newly initialized projection; stage-1/2 handoff unverified, so de-novo adaptation must be labeled |

The existing `FrozenLinearFP32` custom backward returns only input gradients. It must be applied only to frozen base linear weights, not to trainable LoRA branches. Ordinary FP32 low-rank operations may be added alongside the frozen base, with source scaling alpha/r, Kaiming A/zero B initialization and source dropout. Do not use a wrapper that silently discards A/B gradients. Initial A gradient may be zero with B initialized zero; require active B on step 1 and A/B changes after step 2, rather than asserting all tensors change immediately.

Avoid blindly calling the source int8-training preparation on a nonquantized CPU model: casting the entire 7B base to FP32 is incompatible with the intended 24-GiB memory budget. Any replacement loader or precision policy is a disclosed execution adaptation. No installation, new weights, cloud spend or source editing is needed for the proposed gate.

## Ordered remaining work

1. Complete the frozen 256-image TRAIN automatic-proposal census without changing thresholds or cherry-picking. Bind its receipt and state whether natural local inputs exist.
2. Implement a versioned trainable-route adapter with a complete named-parameter inventory. Check formula/zero-initialized LoRA identity, then two actual TRAIN updates with source dropout and finite gradient/module deltas. Start from fresh initialization, not disposable projector gates. Keep a hard memory/time guard and record unsupported components rather than claim full equivalence.
3. Separate proposal/image feature computation from LM execution if required for RAM, but preserve gradients for every declared trainable component. Frozen feature caching is only valid for components explicitly frozen in the prospective protocol; it cannot preserve EVA LoRA training by assertion.
4. Verify frozen state invariance, compact full-state reconstruction and fresh-process exact adapted outputs. Verify public original-image/question inference with no answer/annotation ROI input, including actual free-generation behavior and answer normalization. Teacher-forced loss is not public QA evaluation.
5. Freeze a finite, resource-measured pilot with explicit de-novo projection initialization, source-policy departures, data roles, matched controls, token/generation limits, optimizer/schedule, checkpoint rule and quality endpoint before running it. Do not launch a 1,432-record generative evaluation using old CPU FP16 timings or equate a projector-only exercise with native FAVP completion.

If full source trainability cannot fit available memory, close the hardware limitation explicitly and run only a separately labeled bounded adaptation; do not silently remove modules or let FAVP block independent Lens implementation indefinitely. Final reporting must distinguish executed native assets, adapted training, unavailable stage provenance and unsupported original-method claims.
