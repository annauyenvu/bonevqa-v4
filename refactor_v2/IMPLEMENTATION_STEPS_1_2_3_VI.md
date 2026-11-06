# Báo cáo thực hiện bước 1–2–3

## Phạm vi và nguyên tắc

Mã thí nghiệm lịch sử trong `exp4_multiseed/` và
`mandatory_experiments_20261011/` không bị sửa, nhằm giữ nguyên hash và khả
năng tái lập kết quả đã báo cáo. Bản sửa nằm độc lập trong `refactor_v2/`.

## Bước 1 — protocol, split, checkpoint và fail-closed

Đã thực hiện:

- Nạp đúng split được yêu cầu; không còn fallback `val -> test`.
- Không còn fallback im lặng sang dữ liệu giả. Dữ liệu giả chỉ dùng khi truyền
  cờ `--smoke_fake`.
- Sampling theo toàn bộ group ảnh/study thay vì `rows[:N]`; một ảnh không bị
  cắt rời các câu hỏi giữa sampling.
- Gate chống overlap ưu tiên `patient_id -> study_id -> component_id ->
  image_id`.
- Dataset `fracatlas_component_clean` đọc trực tiếp các manifest đã niêm phong.
- Answer vocabulary được fit duy nhất từ các record train đã chọn. Preflight
  hiện tạo vocabulary 87 đáp án, thay vì tái sử dụng vocabulary lịch sử 127.
- Checkpoint v2 chứa schema, architecture hash, vocabulary hash,
  preprocessing version và tensor shapes. Sai/missing/unexpected shape gây lỗi;
  ngoại lệ transfer-learning phải khai báo bằng prefix trong config.
- API không còn tự động trả lời bằng `MockEngine`. Mock chỉ bật được bằng
  `BONEVQA_ALLOW_MOCK=1`. Segmenter thiếu cũng không được thay bằng ROI ellipse
  heuristic.
- Train/eval/app dùng chung EXIF transpose, RGB conversion và letterbox giữ tỉ
  lệ ảnh.
- Detector v2 nhận cả ảnh dương và âm, không biến ảnh âm thành full-image
  positive box, đồng thời báo negative specificity/FPR.

Kết quả preflight split:

| Split | QA được chọn | Nhóm ảnh |
|---|---:|---:|
| Train | 5.999 | 1.332 |
| Validation | 300 | 68 |
| Test | 2.443 | 550 |

Không có image/component crossing giữa ba split. Train có 5.999 thay vì 6.000
QA vì sampling giữ nguyên toàn bộ nhóm câu hỏi của mỗi ảnh.

Preflight detector trên cùng clean split:

| Split | Tổng ảnh | Dương | Âm |
|---|---:|---:|---:|
| Train | 2.890 | 497 | 2.393 |
| Validation | 643 | 103 | 540 |
| Test | 550 | 119 | 431 |

## Bước 2 — routing, fusion và loss

Đã thực hiện:

- Baseline multimodal luôn tồn tại. V/L/P không còn thay thế bottleneck.
- V và L đi qua `ResidualPromptMixer`; P đi qua residual trong
  `MultimodalFusion`.
- Các gate `gamma_visual`, `gamma_lens`, `gamma_latent` khởi tạo đúng bằng 0.
  Khi bật module, mô hình vì vậy bắt đầu tại đúng baseline.
- V, L và P đều đi vào `x_f`, là biểu diễn dùng chung cho closed head và LLM
  prefix. Không còn tình trạng module chỉ tác động một endpoint.
- P được condition theo image + question. Answer bank chỉ còn là shared semantic
  memory; semantic prior được khởi tạo từ text encoder thay vì embedding ngẫu
  nhiên không mang nghĩa.
- Closed loss chỉ dùng sample `CLOSED`; LLM loss chỉ dùng sample `OPEN`.
- Contrastive loss coi các QA cùng image là multi-positive, không xem chúng là
  negatives của nhau.
- Consistency target của P phụ thuộc chính input, không còn kéo một latent giống
  nhau về nhiều ground-truth answer mâu thuẫn.
- Module tắt không còn được dựng trong model/optimizer, ngoại trừ các khối nền
  thực sự dùng chung.
- Checkpoint selection trong config v2 dùng fracture balanced accuracy; metrics
  xuất riêng sensitivity và specificity.

## Bước 3 — gradient và overfit gates

Kết quả kiểm thử:

- Unit/integration tests: **19/19 passed**.
- Static fatal checks (`E9/F63/F7/F82`): **passed**.
- Overfit 64 mẫu trên routing/fusion độc lập: **5/5 cấu hình passed**.
- Overfit 32 mẫu bằng forward/loss/fusion thật của `BoneVQAModel` với lightweight
  encoders: **None, V, L, P, VLP đều đạt 100%**.
- Preflight với BiomedCLIP thật: forward hữu hạn; gradient gate bước đầu:
  `V=0,007070`, `L=0,005192`, `P=0,006014`.

Kết quả overfit 32 mẫu:

| Cấu hình | Loss đầu | Loss cuối | Accuracy |
|---|---:|---:|---:|
| None | 1,3465 | 0,000310 | 100% |
| V | 1,4041 | 0,000172 | 100% |
| L | 1,4551 | 0,000139 | 100% |
| P | 1,4149 | 0,000100 | 100% |
| VLP | 1,4888 | 0,000087 | 100% |

## Những gì chưa được phép kết luận

Các gate trên chứng minh code mới có đường gradient, có thể overfit và tuân thủ
protocol ở mức triển khai. Chúng **không chứng minh V/L/P cải thiện generalization**.
Muốn thay đổi claim bài báo vẫn phải train lại detector sạch, sau đó chạy các
cấu hình v2 trên clean split qua nhiều seed. Không được trộn kết quả checkpoint
cũ với kiến trúc v2.
