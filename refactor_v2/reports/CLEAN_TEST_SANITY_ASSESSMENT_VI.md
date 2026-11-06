# Đánh giá sanity cho clean-test BoneVQA v2

Ngày đánh giá: 2026-10-15

## Kết quả hợp lệ hiện tại

Nguồn: `eval/results_fracatlas_component_clean_none_seed42_ce_v3_offline_recheck.json`.

- Closed accuracy: 82,56% (1.359 QA).
- Open exact accuracy: 84,23% (1.084 QA).
- Open token recall: 91,58%.
- BLEU-1: 90,13%.
- Overall exact accuracy: 83,30% (2.443 QA).
- Fracture presence tại ngưỡng mặc định: sensitivity 10,48%, specificity 98,20%, balanced accuracy 54,34% (124 dương, 499 âm).
- Fracture ranking sau khi dùng xác suất nhất quán với cross-entropy: AUROC 0,8426.
- Ngưỡng chọn chỉ trên validation: 0,2449385949 trong không gian xác suất CE. Khi khóa ngưỡng này và áp sang test: sensitivity 76,61%, specificity 77,76%, balanced accuracy 77,18%.

## Diễn giải

Closed/open/overall accuracy nằm trong khoảng thường thấy của các benchmark Med-VQA công khai, nhưng không phải so sánh leaderboard hợp lệ vì BoneVQA dùng QA FracAtlas sinh theo template, split component-clean, answer normalization và prevalence riêng. Open exact cao một phần vì tập đáp án ngắn, lặp và theo template; cần image-ablation/grounding để chứng minh mô hình thực sự dùng ảnh.

Sensitivity 10,48% tại ngưỡng mặc định là bất thường và không đạt clinical gate, dù overall/closed accuracy cao. Nguyên nhân trực tiếp là calibration/ngưỡng: AUROC 0,8426 cho thấy score vẫn phân tách hai lớp, nhưng ngưỡng 0,5 tạo 111 false negative trên 124 ca dương. Calibration chỉ bằng validation khôi phục sensitivity 76,61% và specificity 77,76% trên test mà không dùng test để chọn ngưỡng.

Code đã được sửa để dùng xác suất yes/no nhất quán với single-label cross-entropy, lưu `p_yes`, chọn threshold trên validation, rồi khóa cho test. Báo cáo sealed nằm tại `eval/fracture_calibration_none_seed42_ce_v4.json`.

## Artifact không hợp lệ

`eval/results_fracatlas_component_clean_none_seed42_ce_v3.json` không được dùng cho open-answer evidence. Lần chạy này cho open exact 5,44% và 635/1.084 output chứa ký tự ngoài ASCII. Chạy lại với cache/runtime offline cố định cho open exact 84,23% và 0 output ngoài ASCII, trong khi closed metrics giữ nguyên. Đây là lỗi tái lập/runtime của nhánh open generation, không phải clean-test chính thức.

Mọi run sealed tiếp theo phải khóa `HF_HOME`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `BONEVQA_DATA`, `PYTHONPATH` và không tải model trong lúc đánh giá.

## Mốc tham chiếu ngoài

- VQA-RAD ban đầu báo open simple accuracy 25,4% và closed simple accuracy 60,6%; bài báo cũng cảnh báo accuracy/BLEU có thể bị đáp án phổ biến chi phối.
- BiomedCLIP-RadBloomz-7B báo overall 73,2% trên VQA-RAD và 87,5% trên SLAKE.
- MedVInT báo open accuracy 73,7% trên VQA-RAD và 88,2% trên SLAKE.
- FracAtlas gốc báo YOLOv8s localization recall 47,3%, precision 80,7%, mAP@0.5 56,2%. Detector v2 hiện tại thấp hơn ở IoU-hit clean-test, nên V/L có nguy cơ bị giới hạn bởi proposal quality.

Các số ngoài chỉ là sanity range, không phải bằng chứng BoneVQA hơn/kém SOTA vì protocol không đồng nhất.
