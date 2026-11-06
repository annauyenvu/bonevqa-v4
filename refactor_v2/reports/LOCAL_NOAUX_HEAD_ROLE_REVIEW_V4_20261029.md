# Role-specific batch contract repair v4

V3 passed exact original-record equality for localfold0 model-val and wrote its head assessment. It then failed calibration: original quality evaluator explicitly calls canonical=True only for calibration, while model-val/test use mixed QA. Applying mixed QA to calibration was an implementation error in v3, not a training failure.

V4 retains mixed model-val/test batches and canonical calibration batches, exactly matching evaluate() in the locked quality runner. All head/public comparisons remain bitwise exact. Existing localfold0 fitted HEAD_LAST, TRAIN_features, fit receipt and model-val outputs are validated and reused, never overwritten or refitted. Bfold0 remains complete and reused. Other eight heads retain the original fixed ten-epoch TRAIN-only fit. No quality fitting, scientific budget, operating threshold policy, bootstrap or absolute deadline changes.

V1/v2/v3 sources and failures stay intact. V4 supervises one child at a time with timeouts limited by the original coordinator deadline. Postprocess_v4 is authoritative; inspect it before action and never duplicate. Closure requires a new role-aware validator preserving and explaining both failures, verifying all heads/replays/hashes and original deadline; the old finalizer cannot silently treat these failures as absent.
