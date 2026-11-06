# Tổng hợp tác vụ, kết quả và kết luận bước 1–3 — BoneVQA v3

Ngày tổng hợp: 22/10/2026, giờ Việt Nam. Phạm vi là quá trình refinement v3 được ghi trong hồ sơ hiện tại. Các thí nghiệm lịch sử khác được giữ nguyên và không gộp thành kết quả xác nhận của v3. Đây là tổng hợp theo nhóm tác vụ; nhật ký từng lần chạy, kiểm tra và lỗi nằm trong các artifact được dẫn ở cuối.

## Kết luận chính

Đã hoàn tất đợt audit dữ liệu, kiểm tra kỹ thuật, các thử nghiệm có đối chứng và đánh giá native B theo năm fold. Đợt thí nghiệm nhóm đã đóng: không còn epoch hay lượt inference/replay đang chờ trong phạm vi này. Kết quả cuối trên4.013ảnh/17.998QA: VQA exact accuracy90,00%, fracture sensitivity55,94%, specificity94,57%, balanced accuracy75,26% (bootstrap95%:73,16–77,05%). BA là trung bình của sensitivity và specificity.

Kết quả xác minh kỹ thuật không chứng minh hiệu quả lâm sàng. Các đối chứng hiện có chưa xác lập ưu thế T/V/L/P hoặc sequence-uniform/rarity loss. Hiệu chuẩn giảm Brier/ECE nhưng làm giảm BA1,74điểm phần trăm, khoảng bootstrap95% của chênh lệch[−2,65;−0,91]. Chưa chọn Final Solution hay phát hành production. Toàn bộ kế hoạch bảy bước chưa được xác nhận hoàn tất; hồ sơ gốc đầy đủ của cả bảy bước không được tái dựng bằng suy đoán.

## Bước 1 — dữ liệu, provenance, nhãn và phân vùng

| Tác vụ đã chạy | Kết quả và phạm vi |
| --- | --- |
| Exact-byte inventory và đọc/hash lại source |100.219đường dẫn,17.514.996.532byte; tiến trình độc lập tái kiểm tra toàn bộ và tái lập số liệu inventory. Đây là số đường dẫn, không phải số ảnh duy nhất |
| Kiểm tra exact overlap FracAtlas–upstream |55phép so sánh được báo cáo cho nguồn/manifest liên quan đều không có SHA256 trùng; không suy ra độc lập bệnh nhân hoặc hết mọi dạng ảnh biến đổi |
| Audit PMC lịch sử |`train_lon` và test có33.252byte-hash chung. Checkpoint lịch sử không rõ training manifest không đủ điều kiện làm bằng chứng độc lập |
| Audit perceptual FracAtlas–ROCO/PMC |Kiểm tra385.771.320cặp tại pHash radius8;87ứng viên, gồm32ROCO+55PMC. Tái kiểm tra độc lập đúng danh sách;120ảnh ứng viên không có byte-exact/correlation≥0,985/normalized-MAE≤0,01. Không phải bảo đảm chống mọi crop/rotation/composite |
| Strict decoding và quarantine |Bộ FracAtlas niêm phong giữ4.013ảnh trong3.920component;70ảnh bị quarantine theo lỗi/nhãn mâu thuẫn và thành viên liên quan. Source/nhãn gốc được giữ |
| Audit nhãn PMC |28.493QA nguồn →28.490QA TRAIN hợp lệ bằng loại đúng3record malformed; giữ nguyên thứ tự/nội dung record còn lại. Không viết lại nhãn hay khẳng định đã adjudicate lâm sàng |
| Chia năm fold |Mỗi fold có TRAIN/model-validation/calibration/outer; không giao known component trong cùng fold. Outer của năm fold bao phủ mỗi ảnh đúng một lần |
| Vocabulary và class weights |Chỉ dùng TRAIN của từng fold; vocabulary lần lượt125/117/115/127/120. Positive weight suy ra từ nhãn TRAIN, không dùng outer |
| Niêm phong dữ liệu/runtime |Seal alignment36artifact, runtime-vocabulary parent35; binding hash của manifest/config/checkpoint và audit độc lập. Các seal cũ được giữ |
| Detector partition parity |Đối chiếu partition detector với VQA, có mapping vai trò rõ ràng. Metadata khớp không có nghĩa đã train đủ năm detector hoặc export đầy đủ proposal cho mọi fold |
| Nguồn phụ |VQA-RAD chỉ là nguồn bổ sung; split QA lịch sử chia sẻ ảnh nên không dùng như test độc lập theo ảnh |

Fold0 có2.568ảnh/11.502QA TRAIN,321ảnh/1.432QA model-validation,321ảnh/1.432QA calibration và803ảnh/3.632QA outer. Toàn bộ outer có715ảnh nhãn gãy và3.298ảnh nhãn không gãy theo existing ground truth.

Kết luận bước1: hoàn tất audit và niêm phong cho phạm vi dữ liệu development đã khai báo. Known component là nhóm trùng/quan hệ đã biết, không phải patient ID đã được xác minh. FracAtlas đã được xem xét trong development; không được đổi tên thành untouched independent test. Chưa biết mức overlap của dữ liệu pretraining gốc của backbone.

## Bước 2 — kiến trúc, routing, loss và giao thức

Đã sửa/kiểm tra đường nạp split và checkpoint theo hướng fail-closed: không fallback validation sang test, không dùng dữ liệu giả im lặng, kiểm tra schema/config/vocabulary/tensor-shape, giữ nguyên group ảnh khi sampling và không thêm câu trả lời từ validation vào vocabulary.

Pipeline native B dùng ảnh gốc đã chuẩn hóa EXIF/RGB/letterbox224px và câu hỏi thô; gồm BiomedCLIP/PubMedBERT, visual LoRA, QFormer/prefix bridge tới Qwen, fusion, closed-answer head và fracture-presence head. Không truyền GT answer/type/region/annotation ROI vào public prediction.

| Thành phần | Việc đã audit/khóa | Kết luận phạm vi |
| --- | --- | --- |
| B |Khởi tạo chung, ảnh/câu hỏi, answer heads và bridge |Được dùng làm baseline nghiên cứu; chưa đủ cho production |
| V |Local view từ detector tự động, route/fusion và empty-proposal behavior |Gradient/wiring có hoạt động; hiệu quả semantic chưa được xác lập |
| L |Alignment trong TRAIN; original-image inference theo recipe |Kiểm tra teacher/control và phase budget; không xác lập lợi ích so với controls |
| P |GM+LF, answer bank TRAIN-only, residual gate, random-bank/unconditioned/shuffled controls, save/reload |Kết nối kỹ thuật đạt; không có ưu thế semantic đáng tin cậy. Không có PF graph |
| T |Raw question, template semantic, irrelevant token-matched và paraphrase controls |Không xác lập ưu thế semantic của template |
| Đối chứng |Cùng initialization/sampler/budget trong từng so sánh phù hợp |Equal steps không được gọi là equal FLOPs; một số so sánh với B chỉ mang tính mô tả |

Các loss đã rà soát: CLOSED cross-entropy; OPEN token cross-entropy; BCE cho câu hỏi canonical fracture presence; answer-type loss; image-group multi-positive contrastive; các term GM/Lens theo recipe. Fracture head không được route bởi mọi câu hỏi chỉ vì chứa từ “fracture”. Positive weight chỉ lấy từ TRAIN. GT answer embeddings chỉ phục vụ loss trong TRAIN.

Đã tách stage1 ROCO caption alignment để caption CE tạo gradient cho QFormer/prefix; không để recipe FracAtlas component batching hoặc mapping calibration/test bị kế thừa sai vào ROCO/PMC.23candidate recipe đã qua audit cấu trúc. Đây không phải23thí nghiệm chất lượng đã train hoàn chỉnh.

Một số điểm về tái lập nguyên bản LaPA/token aggregation còn là giới hạn diễn giải trong loss review; không tuyên bố toàn bộ implementation là reproduction tuyệt đối chỉ từ gate kỹ thuật. Các quy tắc snapshot cũ, như sensitivity-target calibration, không thay thế giao thức grouped cuối đã đóng băng riêng: estimator cố định và threshold0,5.

Kết luận bước2: các route và recipe CPU đã khai báo có bằng chứng kỹ thuật và provenance. Chưa có bằng chứng cho cấu hình Full V+L+P ưu việt hoặc một loss mới tốt hơn baseline.

## Bước 3 — kiểm tra kỹ thuật, train và thực nghiệm chất lượng

### 3.1 Kiểm tra kỹ thuật và runtime

- Regression suite tại technical seal:104/104passed.190artifact src/config/test của snapshot gốc được hash-verify nguyên vẹn. Các lần19/19,62/63/64tests trước đó là mốc lịch sử, không cộng thành số test mới.
- Bridge real-data smoke: QFormer55/55tensor và prefix4/4có gradient/update trong phạm vi đã thử; VisualLoRA24/24update. Swap ảnh làm đổi logits, xác nhận đường ảnh→answer hoạt động trong fixture; không chứng minh grounding semantic.
- CPU reference epoch-boundary resume:259trainable tensors, AdamW và scheduler khớp chính xác giữa interrupted/continuous trong pilot8TRAIN+4validation. Không mở rộng kết luận thành mid-epoch/MPS/full-training resume.
- Các MPS resume/checkpointing pilot không đạt exact parity; các thất bại được giữ. Full-MPS zero-update gate có hai OPEN answer khác CPU nên không được dùng thay reference cho quality run.
- Tạo common initialization cho B/V/L/P/T/VP và năm PREtask checkpoint theo vocabulary từng fold; kiểm tra hash/strict reload và fresh-process public prediction. PREtask là zero-QA initialization, không phải mô hình đã train VQA.
- Sampler/component-shuffle, gate-zero, bank/prior roundtrip, intervention route, closed/OPEN backward và bounded overfit/capacity đã được kiểm tra trong các fixture tương ứng.
- Benchmark CPU8/10/12threads:8nhanh nhất trong fixture. Full MPS có speed observation≈1,53× nhưng fail OPEN parity. Hybrid vision-MPS/LLM-CPU≈1,13×, giảm11,39%thời gian batch trong tiny fixture; chưa được nâng thành full-fold quality runtime. Các quality run cuối dùng offlineCPU8float32.
- Lỗi OpenMP shared memory, khởi tạo/dataset-path/schema, chênh gradient checkpointing, metric key và cosine float32 đều được lưu. Các run thất bại không được báo thành thành công; chỉ dùng recovery/version có audit riêng.

### 3.2 Pretraining và chẩn đoán upstream

| Nhóm chạy | Kết quả | Kết luận |
| --- | --- | --- |
| ROCO caption alignment mới |2epochs;19.556TRAIN caption;9.903model-validation. Validation loss2,3638→2,2985; bridge/hash/reload đạt. Timer driver ghi23.321giây |Dùng làm upstream reference. Loss giảm không chứng minh ưu thế clinical/VQA |
| PMC generative alignment từ ROCO mới |2epochs,28.490TRAIN/1.500model-validation QA; loss2,7284→2,6885; OPEN exact8,13%→8,27%(124/1500). Timer driver20.664giây |Artifact/provenance đạt nhưng quality gate không đạt; không dùng PMC này làm initialization downstream cuối |
| PMC input/context audit |Rà option/context, flagged517record cần xem xét; candidate raw-question filter27.740retained/750quarantined TRAIN-only |Flag không đồng nghĩa nhãn lâm sàng sai; không biến candidate filtering thành ground truth mới |
| PMC answer-only vs original objective |Pilot128QA: cả hai4/128correct-image,1/128wrong-image;127/128answer giống nhau |Bỏ contrastive không đem lại lợi ích quan sát; không đổi loss chỉ vì total loss thấp hơn |
| PMC TRAIN memorization diagnostic |16/16TRAIN sau100updates |Có khả năng fit fixture; không giải quyết quality/generalization thấp |

Các timer ở đây là giá trị ghi nhận của từng driver; không cộng chúng thành tổng thời gian lịch từ lúc bắt đầu dự án vì không bao phủ mọi audit, thời gian chờ và các run phụ.

### 3.3 Baseline, seed và prompt screens

| Thực nghiệm | Kết quả đã ghi nhận | Kết luận |
| --- | --- | --- |
| B fresh vs ROCO pilot nhỏ |Cùng76steps:ROCO72,54%QA vs fresh69,01%; always-no75,35%QA nhưng sensitivity0 |Accuracy template không đủ để chọn mô hình |
| B mở rộng TRAIN ở fold0 |Fresh/ROCO clinical AUROC≈0,8835/0,8827;ROCO QA113/142, OPEN50/64 |Có image dependence trong interventions; không chứng minh ROCO thắng |
| B seed42/43/44 |Fresh meanAUROC0,8807, meanQA77,46%; ROCO0,8845,78,40% |Ổn định trong cohort nhỏ đã xem; routing-head init khác giữa hai nhánh nên không cô lập causal effect của ROCO |
| P vs random-bank/unconditioned |Clinical AUROC0,88397/0,88364/0,88390; QA115/142,114/142,115/142. P và unconditioned có cùng mọi QA prediction |Chưa có bằng chứng lợi ích GM/LF semantic; không mở rộng P hoặc chọn Final Solution |
| V real vs location placebo |Clinical321AUROC0,88510vs0,88570; BA80,08%vs81,34%; paired ranking intervals chứa0 |Không xác lập semantic benefit từ vị trí proposal thật |
| L real vs original teacher/location placebo |AUROC0,82795vs0,83054/0,82955; BA76,56%vs76,86%/76,30% |Không tốt hơn controls đã khai báo |
| T semantic vs irrelevant/paraphrase |AUROC0,88424/0,88417/0,88404; paired AUROC/AP intervals chứa0 |Không xác lập semantic template benefit |
| Detector/VL wiring và proposal provenance |Train-partition/automatic-box route, reload và controls được kiểm tra; V nhận automatic boxes, L inference ảnh/câu hỏi |Proposal coverage có phạm vi riêng; không chứng nhận đủ năm full-fold V/L pipelines |

Năm V/L arms bounded đã hoàn tất296updates/arm,1.480updates tổng theo completion audit. P và T có gates/pilot/controls riêng. Không được so trực tiếp mọi con số trong bảng như một leaderboard đồng ngân sách: cohort, phase, đầu vào phụ, initialization và parameter/view budget khác nhau giữa các nhóm. OPEN0 trên cohort chỉ có câu hỏi yes/no là trường rỗng, không phải kết quả OPEN thực đo.

### 3.4 Loss screens và các diagnostic tiếp theo

| Tác vụ | Kết quả | Kết luận |
| --- | --- | --- |
| Native/uniform/rarity bounded LAST296 |QA113/142,112/142,114/142; rarity−B interval QA[−2,23;+3,82]pp. Cùng clinical TP53/FN4/TN177/FP87 |Một câu đúng thêm không xác lập cải thiện |
| Đánh giá lại toàn bộ1.432model-validation QA của LAST296 |B78,77%,uniform79,82%,rarity78,91%; OPEN77,07%,80,10%,77,07% |Uniform có tín hiệu mô tả ở pilot; chưa phải xác nhận độc lập |
| TRAIN token budget/routing/errors |Kiểm tra truncation/target decode, rare/unseen answers, existential routing và wrong-image controls |Không có căn cứ coi token limit hoặc một route hỏng là nguyên nhân duy nhất |
| Rare-TRAIN capacity |16record/16component,64updates mỗi arm; native và answer-only đều16/16 |Model fit được fixture; không đủ căn cứ bỏ auxiliary losses |
| Full-TRAIN native vs uniform |Cùng2epochs/1.438updates/11.502TRAIN QA. NativeQA90,08%,OPEN88,69%;uniform89,94%,88,06%. Sensitivity cả hai64,91%. Mọi10paired CI được báo cáo chứa0 |Tín hiệu pilot uniform không được xác nhận ở full TRAIN; không chứng minh superior/equivalent |
| TRAIN canonical diagnosis |2.568ảnh: native sensitivity68,56%(144FN),uniform67,25%(150FN) |Sensitivity đã thấp trên TRAIN; không chỉ là unseen-validation gap |
| Fracture/shared-gradient probes |Manual BCE/CE/LM identities, selected gradients, routing và unchanged-weight checks; tiny mixed fixture visual gradients cùng chiều, fusion gần trực giao |Không chứng minh LM-gradient conflict là nguyên nhân; không có cơ sở tự động gradient surgery/loss deletion |
| Frozen-feature fracture head |Cache2.568TRAIN feature được replay; fixed410updates,10epochs head-only. Modelval sensitivity64,91→84,21%,specificity93,18→83,71%,BA79,05→83,96%,VQA90,08→89,11% |Đánh đổi thêm11TP với25FP; BA difference CI[−0,33;+10,39]pp chứa0, calibration tệ hơn; không promote |

Full-TRAIN native mất4.003giây(66m43s),uniform5.296giây(88m16s); đánh giá/replay pair≈17m51s. Frozen head chỉ cập nhật4head tensors; timer fit cache nội bộ không đại diện tốc độ full-model training.

### 3.5 Năm fold native B và đánh giá outer cuối

Giao thức được đóng băng: seed42 duy nhất,2epochs/fold, LAST cuối epoch2, original image/raw question, T/V/L/Poff. Model-validation chỉ là diagnostic; không chọn lại epoch tốt nhất. Fold0 reuse lịch sử được audit tương thích và công khai giới hạn. Bốn fold mới thực hiện5.768optimizer updates, timer worker cộng25.588giây(7h06m). Không có thử thêm seed, threshold hay budget từ outer kết quả.

Hiệu chuẩn mỗi fold dùng canonical calibration riêng, fixed LogisticRegression(C1,lbfgs,max_iter1000,tol1e-8,seed42,intercept), một feature logit của probability đã clip; threshold0,5giữ cố định. Outer mỗi ảnh xuất hiện đúng một lần. Tất cả prediction, probability, fit coefficients và metric được replay trong process mới; independent audits kiểm tra artifact/role/order/fit/confusion/ranking metrics.

| Metric gộp outer | Raw native B | Calibrated |
| --- | ---: | ---: |
| VQA exact accuracy |90,00%|90,17%|
| CLOSED exact |90,91%|91,21%|
| OPEN exact |88,85%|88,85%|
| Fracture sensitivity |55,94%|50,35%|
| Specificity |94,57%|96,69%|
| Balanced accuracy |75,26%|73,52%|
| AUROC |0,88399|0,88028|
| Average precision |0,71627|0,70479|
| Brier↓ |0,09562|0,08891|
| ECE10↓ |0,05728|0,02875|
| TP/FN/TN/FP |400/315/3119/179|360/355/3189/109|

| Fold | BA raw | BA calibrated |
| --- | ---: | ---: |
|0, historical reuse|72,66%|71,50%|
|1|74,27%|71,49%|
|2|79,24%|76,27%|
|3|73,66%|73,25%|
|4|76,46%|75,10%|

Bootstrap2.000lần, seed20261021, resample known components trong từng outer fold; dùng cùng indices cho raw/calibrated và giữ cố định model/calibrator. RawBA95%interval73,16–77,05%; calibrated71,44–75,38%; calibrated−raw−1,74pp với interval[−2,65;−0,91]. Tất cả2.000primary draws hợp lệ. Audit độc lập tái dựng mọi primary sensitivity/specificity/BA draw từ component confusion counts và kiểm tra mọi percentile/count summary; toàn bộ phân tích được fresh-process replay. Không khẳng định mỗi ranking draw được tái tính bằng thuật toán thứ hai.

Outer inference/replay mất4.701,25giây(78m21s). Bootstrap+replay49,36giây.32artifact kết quả/báo cáo được closure-seal;20metric gộp được đối chiếu thêm với metric implementation.190artifact kỹ thuật gốc không thay đổi.

## Kết luận khoa học và trạng thái

1. Pipeline có bằng chứng kỹ thuật, provenance và khả năng tái lập trong phạm vi CPU/reference đã thử. Các thất bại/recovery được giữ, không xóa khỏi lịch sử.
2. Không xác lập ưu thế T/V/L/P hoặc loss mới từ các screens hiện có. Gradient/update khác0chứng minh connectivity, không chứng minh semantic utility.
3. ROCO mới là upstream reference có lineage kiểm tra được. Chưa cô lập được lợi ích causal/clinical của ROCO; PMC mới không đạt quality gate và checkpoint PMC lịch sử bị loại khỏi independent-evidence claims.
4. VQA accuracy90%không tương đương khả năng phát hiện gãy tốt: raw model bỏ sót315/715ảnh dương theo existing labels, tương ứng44,06%.
5. Hiệu chuẩn đổi40true positives thànhfalse negatives, đồng thời giảm70false positives. Nó cải thiện Brier/ECE nhưng làm xấu primary BA; không dùng tăng overallQA0,17ppđể biện minh chọn calibrated production model.
6. Đợt grouped experiment đã hoàn tất. RawB chỉ là reference nghiên cứu tạm thời; chưa có Final Solution/production release. Chưa xác minh untouched independent clinical cohort, patient grouping hay robustness của thí nghiệm cuối qua training seeds.
7. Full nguyên bản kế hoạch bảy bước chưa được chứng nhận hoàn thành. Phần kết quả âm vẫn là thí nghiệm hoàn chỉnh; việc mở một nghiên cứu cải tiến mới cần thiết kế đóng băng và công khai rằng các outer kết quả này đã được xem. Không có train còn chạy để chờ accuracy tự tăng.

## Hồ sơ và nguồn để truy vết toàn bộ tác vụ

- [Nhật ký chi tiết bước2–3 và các failure/recovery lịch sử](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/STEP2_3_PROGRESS_V3.md); các `CURRENT_GATE_LEDGER_V3_1.json` đến `_70.json` ghi các mốc tiếp theo.
- [Data protocol](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/DATA_PROTOCOL_STEP1_V3.md), [fresh-byte/provenance outcome](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/FRESH_BYTE_AND_UPSTREAM_PROVENANCE_OUTCOME_V3.md), [perceptual scope](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/FRACATLAS_UPSTREAM_PERCEPTUAL_SCOPE_V3.md).
- [Architecture/loss/runtime technical snapshot](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/ARCHITECTURE_LOSS_RUNTIME_PROTOCOL_V3_REVIEW.md), [technical seal](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/STEP3_TECHNICAL_REFERENCE_SEAL_V3.json).
- [ROCO audit](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/roco_alignment_execution_v3/independent_verification.json), [PMC audit](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/pmc_alignment_execution_v3/independent_artifact_verification.json).
- [B seed replication](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/B_matched_seed_replication_v3/interpretation.json), [P controls](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/P_target_semantic_pilot_v3/interpretation.json), [V clinical screen](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/V_full321_development_v3_1/report.json), [L clinical screen](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/L_full321_development_v3_1/report.json), [T evaluation](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/T_independent_quality_current_v3_3/report.json).
- [Loss screen](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/B_SEQUENCE_LOSS_SCREEN_OUTCOME_V3.md), [full-TRAIN comparison](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/FULLFOLD0_FIXED_LAST_OUTCOME_V3.md), [TRAIN fit](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/FULLFOLD0_TRAIN_FIT_OUTCOME_V3.md), [head pilot](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/FROZEN_HEAD_PILOT_OUTCOME_V3.md).
- [Báo cáo outer cuối](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/GROUPED_OUTER_RESULTS_V3_1.md), [kết quả/interval/draw values](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/grouped_outer_bootstrap_v3_1/results.json), [closure seal32artifact](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/GROUPED_EXPERIMENT_CLOSURE_SEAL_V3_1.json), [exit review mới nhất](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/STEP3_EXIT_REVIEW_V3_4.md).

Các câu “pending”, “not released”, “next” trong report cũ mô tả thời điểm viết; đọc cùng audit/recovery/ledger mới tương ứng. Không sửa lại lịch sử để biến failed run thành successful run. Báo cáo paper Final Solution cũ không thay thế kết luận v3 hiện tại.
