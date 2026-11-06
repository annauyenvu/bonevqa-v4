# Kiểm tra đánh giá, rò rỉ và sensitivity — 22/10/2026

Kết luận: kiểm tra mới không phát hiện rò rỉ trực tiếp giữa bốn vai trò dữ liệu trong từng fold, hoặc sai công thức sensitivity trong kết quả đã lưu. Hiệu chuẩn Platt được fit lại bằng đúng tập calibration và thông số đã khóa, tái lập hệ số và xác suất cũ. Nó giảm sensitivity ở cả năm fold. Sensitivity thấp tồn tại trước hiệu chuẩn và không thể giải thích chỉ bằng những dự đoán sát ngưỡng.

Phạm vi: phân tích hậu nghiệm trên thí nghiệm native B seed42, hai epoch, LAST cố định; không train lại, không chọn threshold bằng outer labels, không thay checkpoint hay kết quả đã niêm phong. Thời gian chương trình kiểm tra byte/refit/bootstrap/phân nhóm: 1,843 giây trên dữ liệu và dự đoán đã có; không phải thời gian inference hoặc training.

## Kiểm tra logic và rò rỉ

- 32 artifact của closure seal vẫn khớp SHA256. Kiểm tra hash checkpoint từng fold và manifest từng vai trò.
- Đọc và hash mới byte của 4.013 ảnh. Không giao nhau theo image ID, known component hoặc SHA256 giữa TRAIN / model-validation / calibration / outer-test trong cùng fold.
- Outer có đúng 4.013 image ID xuất hiện một lần, thuộc 3.920 known components. Một ảnh có thể thuộc TRAIN của mô hình fold khác: đó là cách cross-validation hoạt động, không phải rò rỉ vào mô hình đánh giá chính ảnh ấy.
- Mỗi ảnh có đúng một câu canonical `Is there a fracture in this X-ray?`; answer yes/no khớp nhãn `fractured` trong manifest. Đối chiếu nhãn của prediction với manifest. Việc này kiểm tra tính nhất quán dữ liệu, không thay thế đọc phim độc lập để xác minh nhãn gốc.
- Sensitivity = TP/(TP+FN), specificity = TN/(TN+FP), BA là trung bình hai đại lượng; tính trên canonical image-level, không nhân bản theo các câu QA khác. Tính lại sensitivity/specificity/BA/AUROC/AP/Brier từng fold khớp kết quả lưu trong sai số 1e-12.
- Driver fit calibration trước outer inference, chỉ dùng canonical calibration records. Model predict nhận `images`, `questions`; GT answer, answer type, region, boxes không được chuyển vào lời gọi predict. B tắt T/V/L/P và region prompt. Không có optimizer update khi đánh giá.
- Mã TRAIN xây vocab từ TRAIN và yêu cầu sealed vocab bằng đúng tập đáp án TRAIN đã chuẩn hóa; positive weight cũng lấy từ canonical TRAIN (2110/458 hoặc 2111/458). Không lấy tần suất nhãn outer để đặt trọng số.
- Giữ giới hạn chứng cứ: known components không phải patient ID đã xác minh; FracAtlas đã được xem trong quá trình phát triển; PMC train_lon/test nhiễm chéo không phải bằng chứng độc lập. Kiểm tra byte mới này không phải một lượt kiểm tra perceptual/upstream mới. Không được tuyên bố loại trừ mọi rò rỉ tiềm ẩn hoặc có validation lâm sàng độc lập.

## Hiệu chuẩn đã thực hiện lại

Mỗi fold fit LogisticRegression(C=1, lbfgs, max_iter=1000, tol=1e-8, seed42, intercept=True, không class weighting) lên logit của xác suất clip [1e-6, 1−1e-6]. Calibration có 321 ảnh (fold4:322), mỗi fold chỉ có 57 ảnh dương tính. Cả năm fit hội tụ, hệ số khớp bản đã lưu. Áp dụng lại transform lên outer scores khớp xác suất cũ; không fit bằng outer scores/labels.

| Fold | TP / 143 raw | Sensitivity raw | CI95% component raw | Sensitivity sau calibration | Ngưỡng raw tương đương calibrated 0,5 |
|---|---:|---:|---:|---:|---:|
| 0 | 75 | 52,45% | 43,80–60,99% | 47,55% | 0,7380 |
| 1 | 77 | 53,85% | 45,39–62,59% | 46,15% | 0,8355 |
| 2 | 86 | 60,14% | 52,03–68,06% | 53,15% | 0,7659 |
| 3 | 77 | 53,85% | 45,59–62,41% | 51,05% | 0,6676 |
| 4 | 85 | 59,44% | 51,03–67,55% | 53,85% | 0,7942 |

CI mới: 2.000 lần lấy mẫu known components trong từng fold, seed 20261022+fold, giữ model và calibrator cố định. Khoảng này không bao gồm bất định do training seed, fit lại calibration hoặc patient grouping chưa biết. Các CI chồng lấn không chứng minh các fold tương đương; cũng không đủ để kết luận một fold thực sự tốt hơn.

Tất cả slope dương (0,469–0,697), intercept âm. Với q=sigmoid(a·logit(p)+b), q≥0,5 tương đương p≥sigmoid(−b/a). Vì vậy threshold calibrated 0,5 thực tế đòi raw probability khoảng 0,668–0,835, cao hơn 0,5 ban đầu. Đây là nguyên nhân trực tiếp của các quyết định dương tính chuyển thành âm tính sau calibration.

Pooled sensitivity giảm 55,94%→50,35%; TP400→360, FN315→355. Specificity tăng 94,57%→96,69%, FP179→109. BA giảm 75,26%→73,52%. Brier tốt hơn 0,09562→0,08891, ECE10 tốt hơn 0,05728→0,02875. Calibration tối ưu độ phù hợp xác suất, không có mục tiêu sensitivity cao; không được xem cải thiện Brier/ECE là cải thiện phát hiện gãy xương.

AUROC/AP không đổi trong từng fold vì transform đơn điệu. Pooled AUROC thay đổi 0,88399→0,88028 do mỗi fold dùng một transform khác nhau, làm đổi thứ hạng giữa các fold; đây không tự động là lỗi metric. Brier không cải thiện đồng loạt: fold2 xấu đi 0,06714→0,07048. Không chọn bật/tắt calibration từng fold theo outer results.

## Sensitivity thấp: điều quan sát được và điều chưa chứng minh

1. Trong 715 ảnh dương tính có 315 FN; 236 FN có p<0,25, chỉ 79 nằm trong [0,25;0,5). Khoảng 74,9% FN ở xa ngưỡng. Hạ ngưỡng một chút không thể khắc phục toàn bộ vấn đề. Các bin chỉ dùng để mô tả, không phải threshold mới được chọn.
2. Tập đánh giá có 17,82% ảnh dương tính. Accuracy VQA 90% gộp nhiều loại câu hỏi và nhiều âm tính, nên không phản ánh độ nhạy phát hiện gãy. Mỗi fold vẫn có đúng 143 dương tính; chênh lệch prevalence không giải thích được khoảng sensitivity 52,45–60,14%.
3. Chẩn đoán TRAIN trước đây của chính native fold0 chỉ đạt sensitivity 314/458=68,56%, bỏ sót144 ảnh TRAIN. Vì vậy vấn đề không chỉ là suy giảm trên dữ liệu chưa thấy. Tuy nhiên chưa tách được đóng góp của biểu diễn, tối ưu, nhãn và ngưỡng.
4. Pilot head-only trước đây trên model-validation tăng sensitivity 64,91%→84,21% nhưng thêm25 FP, làm xấu Brier và VQA; CI chênh lệch BA chứa0. Chứng cứ cho thấy vị trí ranh giới quyết định có ảnh hưởng, chưa chứng minh head mới tốt hơn hoặc đủ giải quyết các FN khó.
5. Kiểm tra gradient trước đây không xác nhận mất đường gradient hoặc xung đột LM là nguyên nhân. Không có căn cứ để xóa loss phụ hay tăng epoch chỉ từ kết quả này.

## Phân nhóm bằng ground truth sẵn có

| Nhóm nhãn hiện có | TP / số ảnh dương tính | Sensitivity raw |
|---|---:|---:|
| Không hardware | 309/618 | 50,00% |
| Có hardware | 91/97 | 93,81% |
| 1 vị trí gãy | 267/544 | 49,08% |
| 2 vị trí gãy | 109/146 | 74,66% |
| Region hand | 212/377 | 56,23% |
| Region leg | 127/212 | 59,91% |
| Region leg and hip | 21/47 | 44,68% |
| Region shoulder | 0/10 | 0,00% |
| View frontal | 175/350 | 50,00% |
| View lateral | 143/241 | 59,34% |
| View oblique | 10/32 | 31,25% |

Đây là phân tích mô tả hậu nghiệm, các nhóm có thể chồng lấn và chứa nhiều yếu tố gây nhiễu. Region mô tả nhãn vùng ảnh, không tự suy ra vị trí gãy mới. Nhóm shoulder chỉ10 ca nên không được ngoại suy tỷ lệ0% sang dân số. Chênh lệch hardware có thể phản ánh tín hiệu dễ nhận, khác biệt ca bệnh hoặc shortcut; chưa có can thiệp/matched analysis để chứng minh shortcut. Không tự gán nhãn “gãy kín đáo”, mức độ nặng hoặc độ khó từ các kết quả này.

## Vì sao dao động giữa fold?

- Mỗi fold có143 dương tính; chênh lệch tốt nhất–thấp nhất chỉ11 TP, tương ứng7,69 điểm phần trăm. Một ca tương ứng0,70 điểm. CI component rộng khoảng16–17 điểm, nên thành phần mẫu có thể đóng góp đáng kể.
- Cơ cấu ca khác nhau: số dương tính có hardware lần lượt14/11/23/26/23; trong nhóm này TP14/11/21/22/23. Không hardware sensitivity lần lượt47,29/50,00/54,17/47,01/51,67%. Có cả biến thiên thành phần nhóm lẫn hiệu năng trong nhóm; không thể quy mọi chênh lệch cho hardware.
- Calibration mỗi fold chỉ57 ca dương tính, với phân bố điểm khác nhau; raw sensitivity trên calibration lần lượt59,65/52,63/50,88/50,88/47,37%. Fold2 outer tốt hơn rõ so với calibration của chính nó. Hệ số và ngưỡng tương đương vì thế khác nhau; đây là mô tả, chưa định lượng bất định hệ số.
- Mỗi fold đổi cả dữ liệu TRAIN lẫn outer. Chỉ một seed; không tách được training variability khỏi case mix hoặc sampling variability. TRAIN giữa các fold chồng lấn nên không coi5 kết quả là5 lần thử hoàn toàn độc lập.
- Thử chuẩn hóa theo đầy đủ region không có đủ support: một số nhóm dương tính hiếm vắng ở fold0–3, kết quả chuẩn hóa để null. Không điền tỷ lệ giả để tạo một con số đã “điều chỉnh”.

## Kết luận vận hành

Giữ kết quả raw là endpoint chính đã khóa, calibration là kết quả phụ có tradeoff xấu về sensitivity/BA. Chưa có lỗi đánh giá được xác nhận cần sửa trong pipeline này. Lưu danh sách315 FN kèm điểm và nhãn có sẵn cho việc rà soát; không đưa metadata đó vào model input.

Nếu xây dựng phiên bản ưu tiên sensitivity, cần một protocol phát triển mới: xác định mục tiêu sensitivity và chi phí FP trước, chọn operating point chỉ trên phần phát triển/calibration thích hợp, rồi đánh giá trên dữ liệu xác nhận chưa bị dùng để quyết định. Outer hiện tại đã phục vụ phân tích này nên không được đổi nhãn thành holdout độc lập. Không tự động triển khai hoặc chọn threshold tối ưu từ bảng outer.

Nguồn mới: [kết quả kiểm tra và315 FN](sensitivity_calibration_audit_v3_1.json), [chương trình tái lập](../experiments/audit_sensitivity_calibration_v3_1.py). Nguồn lịch sử: `GROUPED_OUTER_RESULTS_V3_1.md`, `FULLFOLD0_TRAIN_FIT_OUTCOME_V3.md`, `FROZEN_HEAD_PILOT_OUTCOME_V3.md`. Báo cáo mới không sửa closure seal hoặc chứng nhận production.
