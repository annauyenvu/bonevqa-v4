# Giới hạn đã giảm và còn lại của hệ thống đã khóa

Ngày 2026-11-01. Hệ thống: B + B head-refit + cờ + điểm detector cao nhất, epoch 6, lớp nhất quán, không crop. Mọi số ở đây là hậu kiểm; V4 AP 0,749 của nghiên cứu niêm phong vẫn là kết quả chính. Test: 5 fold, 4013 ảnh, 715 dương, 17998 câu hỏi.

## Giới hạn đã giảm

| Giới hạn cũ | Kiểm tra | Kết quả | Nguồn |
|---|---|---|---|
| Chọn cấu hình có thể lạc quan | Lồng nhau: mỗi fold chọn từ non-test của 4 fold kia | AP 0,823, so với 0,826 đã khóa; trừ V4 [+0,054; +0,093] | `limits.json` |
| Chỉ một seed detector | Giao thức khóa trước (`seed43/PROTOCOL.md`), train lại epoch 3–6 | AP 0,825 so với 0,825 (hiệu [-0,009; +0,008]); trừ V4 [+0,056; +0,097]; QA 91,75% (trừ V4 [+0,64; +1,43] điểm %); 0 mâu thuẫn. Đạt cả 3 tiêu chí | `seed43/results.json` |
| Detector định vị yếu | Trên 715 ảnh dương, box bất kỳ trong top 3, epoch 2 → 6 | IoU ≥ 0,1: 62,4 → 71,3%; ≥ 0,3: 57,5 → 66,0%; ≥ 0,5: 42,8 → 50,6%. Đối chứng ảnh khác: khoảng 3–4%, 1–2%, < 1% | `limits.json` |
| Đề xuất sai rất khác nhau giữa fold | Ảnh âm có đề xuất sai, theo fold | 16,7–24,1%; detector cũ 3,5–57% | `limits.json` |
| Hiệu chuẩn chưa rõ | Trên test gộp | ECE 0,012; độ dốc 1,000; p trung bình 0,176 so với tỉ lệ 0,178 | `limits.json` |
| Câu trả lời mâu thuẫn nhau | Lớp nhất quán | 0 mâu thuẫn; V4 119, B 207 | `test_results.json` |
| Khó tái lập | Bộ chấm độc lập chỉ dùng thư viện chuẩn | AP 0,825487; QA 0,916657; 0 mâu thuẫn | `experiments/v4_stdlib_audit_v1.py` |
| Khó triển khai lại | Detector chạy mới + `stack_params.json` trên 150 ảnh test | Sai khác lớn nhất 8,9e-16 (logit B lấy từ dự đoán đã lưu, không chạy trực tiếp B) | `experiments/v4_locked_system_v1.py` |

AP theo fold, seed 43 / seed 42: 0,829/0,834; 0,797/0,777; 0,863/0,867; 0,808/0,817; 0,851/0,859.

## B train lại trên fold 0

Kế hoạch khóa trước khi đọc test: `seed43/B_RETRAIN_PLAN.md`. Chỉ mô tả, một fold, không đổi hệ thống đã khóa. Số liệu: `seed43/B_retrain_fold0.json`. Stack [B, cờ, điểm detector], không có head-refit vì các lần chạy v5 không có cột này, nên B niêm phong ở đây không phải 0,834 của hệ thống đã khóa. V4 fold 0: AP 0,740; QA 90,06%.

| B | B một mình: AP | AP, detector seed 42 | AP, detector seed 43 | QA, detector seed 42 | QA, detector seed 43 |
|---|---|---|---|---|---|
| Niêm phong | 0,678 | 0,826 | 0,817 | 91,22% | 91,35% |
| v5 seed 42 | 0,676 | 0,822 | 0,812 | 91,69% | 91,88% |
| v5 seed 43 | 0,691 | 0,824 | 0,817 | 91,71% | 91,77% |

Cả 6 cặp có 0 mâu thuẫn. Train lại B đổi AP của stack tối đa 0,005 khi giữ nguyên detector; đổi detector đổi AP 0,007–0,010 khi giữ nguyên B. Mọi cặp cao hơn V4 fold 0 ít nhất 0,072. B seed 43 + detector seed 42 trừ V4: AP [+0,045; +0,128]; QA [+0,58; +2,71] điểm %.

## Giới hạn còn lại

- **Không có kiểm định ngoài.** Trên máy không có dữ liệu hợp lệ: VQA-RAD trống hoặc không độc lập, PMC không phù hợp, không có ảnh MURA.
- **Test không còn mù.** Test đã được đọc ở các bước trước, nên giao thức seed 43 và lựa chọn lồng nhau không biến kết quả thành xác nhận tiến cứu.
- **Seed 43 chỉ đo một phần độ nhạy theo seed.** Hai seed dùng chung điểm khởi đầu epoch 2 và chung dữ liệu chia fold; chỉ epoch 3–6 khác seed.
- **B mới được train lại trên một fold.** Biến thiên của B chỉ đo trên fold 0 (mục dưới); 4 fold còn lại chưa có. Lần chạy đầu của B seed 43 bị dừng ở cập nhật 630/1438 để nhường máy, rồi chạy lại từ đầu đến hết.
- **Câu hỏi Where kém V4:** 48,1% so với 50,6% (n=158). Trên non-test, khi đáp án là có, detector đúng 31 và mô hình đúng 22 trên n=53, cỡ mẫu nhỏ.
- **Lợi ích không chứng minh bằng chứng đặc hiệu tổn thương theo vùng ảnh.** Hệ thống dùng điểm detector, không dùng crop. Định vị ở trên là của detector, không phải cơ chế của mô hình ngôn ngữ.
- **Cải thiện QA nhỏ:** khoảng 1 điểm %. Vùng bị gãy (câu đóng) không tăng: 89,5% so với 89,6%.
