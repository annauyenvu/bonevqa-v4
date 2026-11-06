# Revised data protocol: closure checklist

Status: development data snapshot sealed by `STEP1_DATA_SEAL_V3.json`; 30 artifact hashes verified. Runtime training readiness remains false until step-2/3 gates pass.

Data sources: local ROCO captions, local PMC-VQA QA, FracAtlas source annotations and templated QA. VQA-RAD remains supplementary and its historical QA split shares images. Do not use it as an image-independent test without a separate grouped protocol.

Current upstream train candidates contain 19,556 ROCO captions and 28,493 PMC QA. PMC model-validation contains 1,500 QA grouped by article. All known perceptual links and PMC article groups are propagated when protecting upstream evaluation cohorts. Historical train_lon is prohibited. Historical stage2 PMC checkpoint is excluded because its exact training manifest cannot be established.

FracAtlas development-informed five-fold component evaluation retains 4,013 images in 3,920 components. Quarantine covers strict decoding failures, contradictory labels and their connected component members (70 images). Source files and labels remain unchanged. Every retained image occurs in exactly one outer test fold. Model-selection and calibration subsets are disjoint from training and outer test. Detector training must use the corresponding training partition, with separate checkpoint/proposal hashes per fold.

Preprocessing policy: EXIF canonical orientation; strict decoding for the filtered protocol; no silent truncated-image acceptance. Keep historical tolerant-decoder results as historical pilot evidence only. Augmentation and image normalization must be resolved and checked against pretrained encoder transforms in step 2.

Inference inputs: image and supported question. Automatic V receives detector proposals; L follows original-image-only inference when trained as alignment. Annotation ROI is supervision/audit/upper-bound data. P scope is GM+LF; answer bank uses training vocabulary only. No PF graph or invented clinical ground truth.

Current verification reports establish no known direct perceptual edge, path or article overlap between filtered upstream train and protected evaluation cohorts, and no historical/new FracAtlas component crossing within generated folds. Limits: duplicate search is heuristic; grouping is not proof of patient identity; development has already inspected historical test; base-model pretraining overlap remains unknown.

Closure checks: independent transitive article/image component separation passed; fold vocabularies match train answers; 24 core manifest artifacts reproduced byte-identically; PMC train has 28,490 label-valid QA after quarantining 3 malformed rows, and the filtered test cohort has 49,990 QA after quarantining 10 malformed rows. PMC model-validation has 1,500 label-valid QA. The final seal lists exact artifacts and limitations. Historical candidate counts above describe intermediate artifacts; use only seal-listed final files.

Step 2 must bind loader paths and runtime hash checks, resolve pretrained image transforms, verify offline base-weight loading, and lock architecture/loss/evaluation routing. Detector checkpoints must be retrained per fold; the existing detector is not eligible for all new folds. VQA-RAD is supplementary and not sealed for image-independent evaluation. No clinical effectiveness claim follows from data checks alone.
