# Local research demo

URL: http://127.0.0.1:8013 (localhost only).

User requested a demo while postprocessing continues. Separate FastAPI server `experiments/serve_local_noaux_demo_v1.py`, UI `experiments/local_noaux_demo_v1.html`; immutable historical app and experiment dependencies unchanged. Existing old offline runtime; CPU2, one worker, only one selected VQA model retained in RAM. No installs, model downloads or external uploads. Image bytes are processed in memory and not saved by this server.

Selectable original-head fixed LAST fold0 B or local_real_noaux, not best-fold/model selection. Checkpoint SHA checked against completion receipt before loading. Local uses strict fold0 detector640 loader, .05/top3 and automatic rectangle masks matching the local-token route. No annotation input or invented masks. No head-refit weights or unverified new operating point. UI labels this as research, not clinical use; score is not a clinically calibrated probability. Vietnamese question mapping is limited; recommend canonical English question. Proposal boxes are detector candidates, not validated fracture localization. Batch1 demo may have small floating-point differences versus batch4 evaluation.

Smoke check: /health returned mock=false. Actual local inference on the existing example gay_IMG0002355.jpg returned `no`, p_yes0.1117558852 with two automatic proposals,6.19sec including loading. This is a technical smoke check, NOT an accuracy result or evidence of success (filename suggests a positive example, and the response was negative). Do not choose a replacement image to hide the failure.

Demo exec session34660. Postprocess v4 supervisor73228/child73229 remained running simultaneously at launch. Demo is not a quality run and does not change assessment records. No automatic retraining or selection from interactive feedback. Browser open requested in current Codex panel; direct local URL is also usable.
