# Mixed TRAIN gradient interaction: bounded result

Same6TRAINcomponents: prior2native false negatives,2true-negative controls and2existing OPEN answers with supervised token lengths2/19. Each fixed LAST checkpoint consumed ONE eval-mode forward; native-tokenmean and sequence-uniform LM losses were reconstructed from the SAME logits/labels with the exact helper used by the private TRAIN adapter (all answer weights1). No checkpoint or data selection from modelval, no optimizer updates and no reserved-role inference.

Manual BCE/closed CE agree with frozen forward; actual native LM versus manual tokenmean selected-gradient maximum errors are at most1e-6. The loss objective was not changed in any saved model. Comparisons use gradients on shared parameter tensors, not orthogonal per-row x_f gradient slots.

| Shared parameter / checkpoint | Fracture vs native LM cosine | Fracture vs uniform LM cosine |
| --- | ---: | ---: |
| First visual LoRA-B / native |+0.240442|+0.240155|
| First visual LoRA-B / uniform |+0.222027|+0.221835|
| Fusion image projection / native |+0.011617|+0.011207|
| Fusion image projection / uniform |−0.012034|−0.012116|

These sampled visual gradients are positively aligned; fusion projection gradients are nearly orthogonal with a small negative cosine for the uniform checkpoint. This does NOT establish that LM competition caused the full-TRAIN sensitivity failure, nor rule it out on other parameters/examples/stochastic batches. Sequence-uniform changes LM magnitude much more than direction on these two answers; it is not justified to delete LM, introduce gradient surgery, or increase fracture weighting from this tiny diagnostic alone. QFormer/prefix/LLM LoRA participate in LM but not fracture BCE on this current route; fracture-head parameters receive no LM gradient, as expected.

Preserved v3_1 used float32 norm/dot reductions and produced a few near-collinear cosine values up to1.000035. Its losses/gradients were unchanged, but those cosine statistics were not accepted as precise geometry. v3_2 changes ONLY detached cosine reductions to CPUfloat64, without clamping, model/loss precision changes or assertion relaxation. All losses and recorded gradient norm/max summaries match v3_1 exactly. Float64 cosines are finite and bounded within ordinary machine roundoff. Both versions and their artifacts remain preserved.

All263trainable hashes and original checkpoints remained unchanged. v3_2 recorded losses/gradient summaries/cosines replayed exactly in a fresh process; this is not a comparison of all unpersisted full-gradient arrays. Separate artifact review checked bindings, equivalence to v3_1 and numerical statistic bounds without a neural model. Elapsed19.154237seconds including replay. Protocol SHA7e6d2a0ac5d72e1761108eda146cc9a2c035f0c4e540bcc1e8b7ea6c1acac9d5; report SHA1abee74372d188c4ff452b2368811c957560d52e01628d2e3399ef9bd2fc8f31.

Next discriminating candidate: a fixed-feature, fracture-head-only bounded TRAIN fit with the original head as matched unchanged control. This directly tests whether additional head optimization can exploit the frozen representation, without repeatedly backpropagating through the752M-parameter model or claiming a new loss mechanism. It requires a locked feature/input/label cache, strict baseline score replay, fixed training horizon, and independent candidate replay before execution/evaluation. It is not yet implemented or released. No current accuracy, clinical benefit or step3 completion claim changes.
