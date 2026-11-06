# Đối chiếu kết quả với yêu cầu bảy bước gốc — 23/10/2026

## Kết luận điều hành

Đã thực thi và khép kín chuỗi thí nghiệm được phát hành đến bước 6: pilot, screening cùng cohort, và đánh giá Native B trên năm training seeds. Chưa đủ căn cứ đánh dấu toàn bộ yêu cầu gốc của bước 1–6 đều hoàn tất, và bước 7/Final Solution chưa khép kín. Các từ “complete” trong closure trước đây có phạm vi thí nghiệm cụ thể, không phải chứng nhận toàn bộ kế hoạch.

Pilot/screening cho kết quả âm tính về lợi ích ngữ nghĩa V/L/P so với các control đã chạy. Đây là kết quả hợp lệ, không phải lỗi vì không tìm được module tốt hơn. Không có ứng viên đạt quy tắc đi tiếp, nên bước 6 chỉ đo độ ổn định baseline; không có xác nhận nhiều seed về ưu thế V/L/P hoặc combo.

Nguồn yêu cầu: `/Users/uyen/.codex/attachments/42a609df-6104-442a-b90b-0989f6aefe0f/Văn bản đã dán.txt`, đọc đầy đủ. Tệp này trình bày yêu cầu khoa học nhưng không đánh số riêng từng bước; cách gọi bước 4–7 dưới đây theo bảng người dùng đã cung cấp trong cuộc trò chuyện. Bước 1–3 được nhóm theo protocol/dữ liệu, kiến trúc/loss và technical gates; không tự tạo tiêu chí đạt chất lượng lâm sàng bằng một ngưỡng tùy ý.

## 1. Phạm vi kiểm tra lần này

- Kiểm tra tiến trình: không thấy job train/assessment/supervisor thuộc chuỗi bước 5–6 trong danh sách tiến trình khớp mẫu; không mở lại training/inference.
- Đọc protocol gốc, các review dữ liệu/loss/technical, protocol và closure pilot, screening, thiết kế năm seed và audit đánh giá.
- Băm lại 30 artifact của STEP1_DATA_SEAL_V3; 66 dependency của data_lineage_closure_v3_2; 190 artifact của technical reference; 32 artifact của grouped closure; 9 artifact của step4 closure; 17 artifact của steps56 closure: tất cả khớp. Các tập này có phần giao nhau, không cộng thành số artifact độc lập.
- Băm lại dependency của execution releases bước 4/5 và training/assessment releases bước 6: lần lượt 78/58/74/79 dependency, tất cả khớp.
- Tái tính trực tiếp TP/FN/TN/FP từ prediction records cho cả raw/calibrated của từng seed 42–46: đúng 803 ảnh canonical duy nhất/seed, 143 dương/660 âm, 3.632 QA/seed; sensitivity/specificity khớp metric đã lưu.
- Kiểm tra cách triển khai screening và audit bước 6: chọn fixed LAST, điều kiện AP đã khóa; calibration fit riêng; cùng component bootstrap qua seed/variant; không gộp 803×5 thành 4.015 ảnh độc lập.

Đây là audit artifact, logic và phạm vi kết luận; không phải chạy lại mọi model, mọi test hay toàn bộ inventory ảnh thô. Hash khớp chứng minh artifact không thay đổi, không tự chứng minh mọi giả định khoa học đều đúng. Replay và các kiểm tra full-data trước đó là bằng chứng lịch sử đã niêm phong, không được mô tả thành lượt replay mới trong lần kiểm tra này.

## 2. Trạng thái từng bước

| Bước | Đã có bằng chứng | Đánh giá so với yêu cầu gốc |
| --- | --- | --- |
| 1. Khóa bài toán/protocol/dữ liệu | QA từ GT/metadata, train-only vocab, kiểm tra exact/perceptual duplicate, quarantine, manifests train/model-val/calibration/outer tách component; loại PMC train_lon/checkpoint không đủ provenance | Đạt cho snapshot nghiên cứu development-informed. Không chứng nhận patient independence, không biết toàn bộ pretraining của base model, không có untouched external holdout |
| 2. Kiến trúc và loss | B/V/L/P được định nghĩa; LaPA GM+LF không PF; canonical BCE, CLOSED/OPEN routing, train-only weights; shared GM không bị gọi sai là sample-conditioned; L train-only, V automatic proposals | Đạt nền kỹ thuật cho recipe đã chạy. Chưa khép toàn bộ kiểm chứng phương pháp: P consistency on/off và trained bank-off chưa có quality evidence cùng cohort; một số khác biệt so với tác giả gốc vẫn được khai báo là variant |
| 3. Technical gates/runner | Gradient/update/routing, CPU offline initialization, sampler, reload/replay, scoped resume; releases riêng trước các quality runs; full native B grouped experiment đã hoàn tất | Đạt các gate cần thiết cho các lượt chạy đã phát hành. Không suy rộng thành mọi cấu hình/23 recipe/MPS/production đều được chứng nhận. Các cờ step3_complete=false cũ là snapshot lịch sử và không tự động được sửa bởi việc đã chạy bước sau |
| 4. Pilot có giới hạn | B/V-real/V-placebo 4 epoch, 292 update/arm; learning curves, full output replay, paired bootstrap; L/P có pilot lịch sử và sau đó được bổ sung same-cohort tại bước 5 | Đạt pilot tối thiểu và feasibility. Chưa đạt bằng chứng module hữu ích; grounding/box-quality–FN synthesis theo protocol chưa đầy đủ |
| 5. Screening một seed | 9 arm gồm B, V-real/placebo, P-real/random/unconditioned, L-real/original/placebo trên cùng cohort; fixed LAST, controls cùng phase trong L; selection + bootstrap đã xong | Đạt ma trận screening đã khóa, không ứng viên qua. Không phải đã chạy mọi control yêu cầu trong tài liệu gốc hoặc 23 ablation. B so với L không phase-matched; đối chứng L trong cùng phase mới là so sánh chính |
| 6. Xác nhận nhiều seed | Native B seeds 42–46, 292 update/seed; seed42 reuse công khai; seed43–46 task initialization/schedule mới; calibration/replay/audit/2.000 bootstrap đã xong | Hoàn tất baseline stability trong phạm vi đã khóa. Không phải xác nhận module/combo, không phải xác nhận độc lập hoặc nested selection-confirmation của toàn pipeline |
| 7. Phân tích/contributions/Final Solution | Đã có bootstrap, negative findings, failure analysis theo metadata, các report từng thí nghiệm | Mới hoàn thành một phần. Thiếu tổng hợp thống nhất các yêu cầu còn mở, claim–evidence/contributions cuối cùng và gói Final Solution có giới hạn sử dụng rõ; chưa production |

Không có yêu cầu phải chạy combo nếu không module nào có tín hiệu riêng. Việc không mở VP/V+L/T và không tăng epoch/seed để tìm kết quả dương là phù hợp nguyên tắc đã khóa, không phải tự thân một phần việc còn thiếu.

## 3. Kết quả chính và cách diễn giải đúng

### 3.1. Native B, full TRAIN, năm outer folds — seed42

4.013 ảnh/3.920 component/17.998 QA, mỗi ảnh xuất hiện một lần trong outer evaluation; hai epoch, fixed LAST, V/L/P/T tắt. Các vai trò tách theo component trong từng fold, nhưng dữ liệu đã có lịch sử sử dụng phát triển. Đây là grouped estimation của pipeline cố định, không phải thí nghiệm chọn module trong inner loop rồi xác nhận độc lập ở outer loop.

| Metric | Raw | Calibrated |
| --- | ---: | ---: |
| Fracture AP | 0,71627 | 0,70479 |
| AUROC | 0,88399 | 0,88028 |
| Sensitivity | 55,94% | 50,35% |
| Specificity | 94,57% | 96,69% |
| BA | 75,26% | 73,52% |
| Brier | 0,09562 | 0,08891 |
| ECE10 | 0,05728 | 0,02875 |
| Overall VQA exact | 90,00% | 90,17% |

Calibrated−raw BA −1,74 điểm phần trăm, paired component 95% interval [−2,65; −0,91]. Calibration làm thêm 40 FN và giảm 70 FP. Không lấy overall VQA tăng hoặc Brier giảm làm bằng chứng phát hiện gãy tốt hơn. Pooled AP/AUROC có thể đổi dù từng fold dùng calibration đơn điệu, vì mỗi fold có một phép biến đổi khác nhau.

Nguồn: `GROUPED_OUTER_RESULTS_V3_1.md`, `GROUPED_EXPERIMENT_CLOSURE_SEAL_V3_1.json`.

### 3.2. Pilot/screening cùng cohort nhỏ — seed42

256 TRAIN ảnh/1.164 QA; 321 model-validation ảnh/1.432 QA. B/V dùng lại pilot đã xem, P/L là các lượt bổ sung. Mỗi arm tổng 292 update; L gồm 146 alignment + 146 VQA, không cùng phase exposure với B. Fixed LAST4, không chọn epoch đẹp nhất.

| Arm | Fracture AP trên model-validation |
| --- | ---: |
| B | 0,668688 |
| V-real | 0,669935 |
| V-placebo | 0,674111 |
| P-real | 0,668107 |
| P-random | 0,683351 |
| P-unconditioned | 0,662946 |
| L-real | 0,665445 |
| L-original | 0,676976 |
| L-placebo | 0,657304 |

Quy tắc: module phải hơn B và từng control riêng ít nhất 0,005 AP, tối đa một ứng viên. Không module nào đạt. Các exploratory paired AP intervals đều chứa 0; không dùng chúng để thay quy tắc đã khóa, cũng không kết luận tương đương hay module vô ích ở mọi thiết lập. V-real−placebo AP của pilot −0,004176, 95% interval [−0,016126; +0,003750].

Nguồn: `step4_localization_v3_1/STEP4_EXIT_EVIDENCE.md`, `step56_screening_v3_1/SCREENING_OUTCOME.md`, `screening_selection.json`, `paired_AP_bootstrap.json`.

### 3.3. Năm training seeds của baseline cohort nhỏ

Cùng 256 TRAIN ảnh; assessment 803 ảnh đã được xem trong phát triển. Không so sánh trực tiếp mức AP này với full-TRAIN five-fold ở mục 3.1 để kết luận model tiến/lùi: khác lượng TRAIN, horizon và tập đánh giá.

| Metric, trung bình 5 seed | Raw | Calibrated |
| --- | ---: | ---: |
| AP | 0,51202 ± 0,01694 | 0,51202 ± 0,01694 |
| AUROC | 0,80401 ± 0,01157 | 0,80401 ± 0,01157 |
| Sensitivity | 46,43% ± 19,03 điểm | 26,85% ± 2,35 điểm |
| Specificity | 88,67% | 95,94% |
| BA | 67,55% | 61,40% |
| Brier | 0,14938 | 0,11861 |
| ECE10 | 0,13021 | 0,03935 |

± là sample SD giữa seed, không phải CI. Raw sensitivity biến động 23,08–69,93%; các TP lần lượt 100/58/33/53/88 trên 143 ca dương. Calibrated TP 35/42/36/37/42. Calibrated−raw BA −6,15 điểm, component 95% interval [−8,18; −4,12].

Năm mô hình lặp lại cùng ảnh không tạo năm cohort độc lập. Bootstrap giữ năm model đã fit cố định; không bao phủ mọi training seed hay patient grouping chưa biết. Có bằng chứng score/decision instability và calibration đổi operating point, chưa có bằng chứng nhân quả rằng tăng epoch, đổi loss hay tăng seed sẽ chữa được vấn đề.

Nguồn: `step6_confirmation_v3_1/FIVE_SEED_RESULTS.md`, `independent_analysis.json`, năm `assessment_seed*_B.json`, `steps56_closure.json`.

## 4. Yêu cầu gốc chưa thể đánh dấu hoàn tất

### A. P: còn thiếu phép thử nhân quả được yêu cầu rõ

Đã chạy real/random/unconditioned, có technical bank-off/gate-zero interventions. Không tìm thấy quality checkpoint/result cho separately trained consistency-off hoặc bank-off trong ma trận hiện tại. `configs/control_p_consistency_off_v3.yaml` và structural config audit không phải kết quả thực nghiệm. Vì vậy chưa phân biệt hết tác động answer bank, consistency regularizer và LF theo yêu cầu gốc. Điều này giới hạn kết luận cơ chế P, không đảo ngược việc P-real hiện không qua screening.

### B. Grounding/localization: có bằng chứng, nhưng chưa đủ chuỗi giải thích

Không đúng nếu nói chưa đánh giá detector: so sánh 320/640 đã có, cùng 321 modelval ảnh. Best top5 IoU≥0,5 hit 10/57 lên 28/57; budget top3 của runtime chỉ 26/57. Đây là positive-image hit, không phải lesion recall hay diagnostic sensitivity. Detector vẫn bỏ sót/định vị yếu ở nhiều ca; automatic proposal presence không đồng nghĩa lesion coverage.

Chưa thấy báo cáo hoàn chỉnh nối chất lượng proposal/GT-box overlap, baseline FN và real–placebo gains/losses của pilot mới; chưa khép đầy đủ bảng IoU0,25/0,5, FN coverage và phân tầng chất lượng box theo Stage A. Các gate có gradient/update/residual không thay thế phân tích này. Không từ association hardware/fracture_count suy ra model chắc chắn dùng shortcut dụng cụ.

Nguồn: `detector640_fold0_bounded_quality_v3_1/interpretation.json`, `detector640_top3_binding_v3/report.json`, `STEP4_PROTOCOL_REVIEW_V3_1.md`.

### C. VQA difficulty và question-family reporting: có lịch sử, chưa đồng bộ thí nghiệm mới

Đã có TRAIN-majority và question-only/geometry-only diagnostics cũ; không được gọi là chưa làm bất kỳ baseline nào. Nhưng chúng dùng cohort 261 TRAIN/1173 QA và 142-QA subset, khác pilot/screening hiện tại. Geometry diagnostic còn có detector được train bằng 2.568 ảnh/annotation, không phải pipeline chưa từng nhìn pixel hoặc capacity-matched B.

Các report bước 5–6 hiện báo chủ yếu aggregate/CLOSED/OPEN và canonical fracture, chưa có bảng question-family và majority/question-only/geometry-only trên cùng protocol mới để đối chiếu thống nhất. Có thể bổ sung một phần từ saved predictions/metadata mà không cần train lại model. Không tự gọi majority resubstitution trên TRAIN là held-out baseline.

Nguồn: `geometry_diagnostic_current_v3_1/protocol.json`, `independent_verification.json`, `transform_and_interpretation_audit.json`, `B_fullTRAIN_errors_v3_1/protocol.json`.

### D. Xác nhận và phạm vi claim

Đã có component-disjoint role manifests và full B grouped evaluation; không phát hiện mismatch hash hay mâu thuẫn đếm canonical trong phạm vi kiểm tra lần này. Tuy nhiên không thể kết luận “không còn mọi loại rò rỉ”: known components không phải verified patients; heuristic duplicates không bao phủ mọi crop/composite; pretraining độc quyền không biết đầy đủ; repeated test inspection là development exposure.

Năm-seed study dùng lại fold0 đã xem; không thay thế external compatible holdout hoặc nested component selection-confirmation cho một thiết kế mới. Full five-fold B là fixed-recipe estimation, không chứng minh đã xác nhận toàn bộ B/V/L/P selection pipeline. Không có detector/proposal thực nghiệm đủ năm fold để mở rộng claim sang V/L. Vì không module nào được chọn, không cần tự động chạy năm seed cho chúng chỉ để lấp bảng.

### E. Báo cáo cuối và Final Solution

Có thể đóng một kết quả nghiên cứu âm tính có giới hạn: baseline còn nhiều FN, calibration cải thiện xác suất nhưng không cải thiện detection, V/L/P chưa vượt controls trong ngân sách đã thử. Không bắt buộc tìm được module tốt hơn mới được hoàn thành báo cáo.

Chưa đủ bằng chứng cho “fusion mới hữu ích”, “L giúp original inference ổn định”, hay “hệ lâm sàng sẵn sàng”. Cần bảng contributions phân biệt supported/rejected/unresolved, failure/grounding synthesis và quyết định cấu hình cuối kèm checkpoint/preprocessing/calibration/input contract với giới hạn. Native B có thể là research reference ít phức tạp nhất trong số đã thử, nhưng đó không tự động là một clinical/production Final Solution đã được xác nhận.

## 5. Thứ tự khép kín hợp lý

1. Hoàn thiện bước 7 từ artifact sẵn có: question-family metrics, so sánh controls đúng cohort/phase, calibration trade-off, localization–FN synthesis và claim–evidence table. Không cần chạy lại các job hoàn tất.
2. Ghi rõ các yêu cầu chưa thử: P consistency-off/trained bank-off, các hạn chế confirmation/grounding. Nếu muốn khẳng định cơ chế hoặc thay đổi thiết kế, cần một study mới được khóa trước; không âm thầm biến analysis thành thêm training.
3. Chọn mục tiêu cuối: đóng báo cáo baseline/negative findings trong phạm vi development, hoặc yêu cầu một nghiên cứu cải thiện/xác nhận mới. Không đổi threshold theo assessment, không gọi dữ liệu đã xem là untouched, không suy từ thêm tài nguyên/thời gian ra chắc chắn cải thiện.

**Trả lời ngắn “đạt tới đâu”: đã chạy xong phạm vi bước 5 và nhánh baseline của bước 6; đang ở khâu tổng hợp/khép yêu cầu bước 7, đồng thời còn các mục kiểm chứng gốc chưa đủ ở P, grounding và confirmation. Chưa hoàn thành toàn bộ bảy bước hay production.**
