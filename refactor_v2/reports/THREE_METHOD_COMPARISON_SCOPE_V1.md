# Phạm vi so sánh BoneVQA v3 và ba nhánh tham chiếu

Bản tổng hợp phạm vi sau khi cả ba pilot giới hạn đã đóng. Xem báo cáo bàn giao `THREE_METHOD_BOUNDED_PILOT_SUMMARY_20261027.md` để có diễn giải cuối. Không dùng bảng này để xếp hạng chéo các số đo khác nhau. Các checkpoint/kết quả tham chiếu không tự thay thế hay hợp nhất vào checkpoint BoneVQA v3.

| Nhánh | Phạm vi đã thực hiện | Tập đánh giá / số đo | Kết luận có căn cứ hiện tại |
|---|---|---|---|
| BoneVQA v3 native B cố định | T/V/L/P tắt; grouped outer evaluation; LAST, seed42 | 4.013 ảnh /17.998 QA; raw fracture BA75,26%, AP0,71627; overall QA90,00% | Đây là mốc B đã khóa, không phải mô hình đã bật đầy đủ LaPA/FAVP/Lens. Sensitivity55,94% cho thấy QA90% không đồng nghĩa phát hiện gãy tốt. |
| LaPA native-backbone pilot đã đóng | Native M3AE, image/text frozen, pretrained MM, prompt/head mới; graph409 tái dựng thay graph577;5nhánh,292updates/nhánh | 1.432 QA /314component bootstrap; closed41class head; AP carrier0,660138,GM+LF0,526043,authorPF0,430638 | AuthorPF−GM+LF AP−0,095405; CI[−0,172551;−0,011226]. Không nâng cấp nhánh PF từ kết quả này; không chứng minh LaPA gốc thất bại. |
| FAVP native-asset adaptation đã đóng | Native SAMMed2D/EVA/QFormer/Vicuna, projection random do thiếu stage handoff;64updates/32TRAIN; source SAM không có proposal chấp nhận nên original fallback | 8ảnh model-validation đã dùng trong phát triển; generation strict accuracy PRE0/8→LAST6/8 | LAST trả lời no toàn bộ, bằng TRAIN-majority constant6/8. Chưa có bằng chứng lợi ích điều kiện theo ảnh hoặc nhánh local. Không phải exact original FAVP. |
| Lens native-asset adaptation đã đóng | Native SAMHuge/SmolVLM,1024longedge thay2048, region labels từ CSV hiện có; hai nhánh cùng3views, tắt/bật DCL/InfoNCE;28TRAIN hợp lệ từ32candidates | Cùng8ảnh development model-validation đã dùng cho FAVP; primary strict full-answer EM; secondary leading yes/no được khóa trước | Cả hai nhánh6/8, No toàn bộ, bằng majority no; Lens−CE0, chưa có lợi ích loss-on. Fresh replay/audit/bootstrap/PNGreview/v2closure44artifacts đã qua. Chỉ là so sánh loss route khi cùng view training, chưa phải toàn bộ view mechanism. |

## Những điều không thể suy ra từ số liệu hiện có

- Không so trực tiếp 90% của v3 với75% của FAVP hay AP của LaPA để kết luận kiến trúc tốt hơn: cohort, backbone, pretraining, số cập nhật và loại endpoint khác nhau.
- Không coi nhãn region ảnh là GT segmentation/organ riêng cho mask. Không có annotation ROI ở đường public inference; SAM proposals là tự động, không phải segmentation ground truth.
- Không coi8case tái sử dụng là validation độc lập. Tất cả dữ liệu ở đây development-informed; component không phải định danh bệnh nhân đã xác minh, và interval một seed không phản ánh biến thiên seed/convergence.
- Không đổi threshold/epoch/seed/parser để cứu kết quả sau khi xem assessment. Giữ kết quả âm, truncated/unsupported answers và các lỗi đã lưu.

## Sau khi Lens đóng

Kết quả Lens đã được tổng hợp từ closure/audit/bootstrap, so loss-on−loss-off và baseline majority, không chỉ LAST−PRE. Báo cáo riêng mức độ bám nguồn và điều chỉnh tài nguyên/supervision của từng phương pháp; baseline và các nghiên cứu cũ được giữ nguyên. Nếu sau này nghiên cứu tích hợp BoneVQA với công tắc LaPA/FAVP/Lens, phải phân biệt mô hình được train với module tắt/bật và việc chỉ tắt module ở inference; Lens thường thay đổi loss huấn luyện chứ không có cùng một module inference để tắt. Chưa mở tám nhánh train hay tự tuyên bố mô hình v3 mới.

Nguồn local đã kiểm tra: `GROUPED_OUTER_RESULTS_V3_1.md`, `lapa_native_quality_v1/FINAL_INTERPRETATION.md`, `favp_bounded_quality_v1/FINAL_INTERPRETATION.md`, `lens_bounded_quality_v1/FINAL_INTERPRETATION.md`, `lens_bounded_quality_v1/closure_seal.json`, `LENS_BOUNDED_NATIVE_ASSET_PILOT_PROTOCOL_V1.md`. Không lấy tiến độ/gate làm hiệu quả held-out;8case vẫn là dữ liệu phát triển tái sử dụng.
