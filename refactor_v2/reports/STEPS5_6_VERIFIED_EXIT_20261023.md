# Kết thúc screening và đánh giá năm seed — 23/10/2026

Đã hoàn tất phạm vi bước5–6 đã khóa: screening trên cohort256TRAIN ảnh1164QA và321model-validation ảnh1432QA, sau đó đánh giá baseline seeds42–46 trên cùng803ảnh assessment đã có lịch sử phát triển. Đây không phải xác nhận lâm sàng độc lập hoặc kết quả của mô hình train toàn bộ FracAtlas.

Bước5: sáu nhánh P/L mới đã hoàn tất292updates/nhánh và replay đủ1432output; B/V seed42 được tái sử dụng từ pilot đã audit. Không V/L/P-real nào đạt quy tắc AP vượt B và tất cả control riêng ít nhất0,005. Learning curves và2.000paired component bootstrap đã có; tất cả khoảng AP đối chứng thăm dò chứa0. Không suy ra tương đương hoặc vô ích trong mọi thiết lập.

Bước6: baseline là nhánh duy nhất đi tiếp. Seed42 được tái sử dụng; seeds43–46 có task heads khởi tạo lại theo seed, upstream không-task cố định, lịch QA theo từng seed và292updates/seed. Các gate/reload đạt trước khi training. Năm lần calibration/assessment và năm replay độc lập tiến trình đã khớp chính xác. Audit lại nhãn/order, coefficient calibration, counts và metric đã đạt;2.000component draws dùng cùng mẫu giữa các seed/biến thể, không gộp803ảnh×5 thành4.015ảnh độc lập.

| Metric trung bình năm seed | Raw | Platt, threshold0,5 |
|---|---:|---:|
| AP |0,5120 ±0,0169|0,5120 ±0,0169|
| AUROC |0,8040 ±0,0116|0,8040 ±0,0116|
| Sensitivity |46,43% ±19,03điểm|26,85% ±2,35điểm|
| Specificity |88,67%|95,94%|
| Balanced accuracy |67,55%|61,40%|
| Brier, thấp hơn tốt hơn |0,1494|0,1186|
| ECE10, thấp hơn tốt hơn |0,1302|0,0394|

± là sampleSD giữa năm seed, không phải confidence interval. BA raw có component95%interval63,92–71,14%; chênh lệch calibrated−raw BA là−6,15điểm, interval−8,18đến−4,12điểm. Các interval giữ năm model đã fit cố định, không bao phủ mọi seed hay patient grouping chưa biết.

Kết luận: chất lượng detection còn hạn chế và độ nhạy raw@0,5 dao động đáng kể theo seed; Platt cải thiện độ phù hợp xác suất nhưng giảm sensitivity/BA. Không chọn calibration làm cải thiện phát hiện gãy, không đổi threshold từ assessment. Không có module đủ điều kiện chứng minh contributions ngữ nghĩa hoặc Final Solution. Không suy ra thêm epoch/seed/loss search sẽ sửa được vấn đề.

Đã kiểm tra17hash closure bước5–6,9hash bước4,32hash grouped trước đó và190artifact technical reference: tất cả còn khớp. Không còn job train/assessment/audit khớp mẫu tiến trình kiểm tra. Các pending flags trong file hoàn tất thực thi cũ là lịch sử và được supersede bởi independent_analysis.json và steps56_closure.json.

Nguồn chính: step56_screening_v3_1/SCREENING_OUTCOME.md; step56_screening_v3_1/paired_AP_bootstrap.json; step6_confirmation_v3_1/FIVE_SEED_RESULTS.md; step6_confirmation_v3_1/independent_analysis.json; step6_confirmation_v3_1/steps56_closure.json. Toàn bộ bước7, Final Solution và production chưa được chứng nhận; không tự mở thêm thí nghiệm để tìm kết quả dương tính.
