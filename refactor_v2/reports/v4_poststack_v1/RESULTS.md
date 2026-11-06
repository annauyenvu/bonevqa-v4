# V4 + điểm detector (ghép muộn, hậu kiểm)

Ngày 2026-11-01. Không huấn luyện lại mạng. Không ghi vào thư mục niêm phong `local_noaux_refinement_v1`.

**Phương án khóa:** hồi quy logistic trên logit V4, logit V4 head-refit, cờ có đề xuất và logit điểm detector cao nhất. Chỉ đổi câu "Is there a fracture in this X-ray?".

**Chọn:** 8 ứng viên, cross-fit hai chiều calibration/model-validation, gộp 5 fold (3211 ảnh, 570 ảnh gãy), chọn ứng viên đơn giản nhất cách tốt nhất không quá 0,005 AP. Khóa trong `selection.json` trước khi chạy test trong bước này. Các nhánh V4, B đã từng được đọc trên test ở nghiên cứu trước.

**Đánh giá:** fit lại trên calibration + model-validation của từng fold; ngưỡng từ âm tính ngoài fold, đặc hiệu mục tiêu 0,90; bootstrap 2000 lần theo component trong fold, seed 20261105. Ảnh và component của TRAIN không trùng với ảnh được chấm (0 ở cả 5 fold). 4013 ảnh test, 715 ảnh gãy.

| Bộ chấm | AP | AUROC | Độ nhạy | Độ đặc hiệu | BA | Brier |
|---|---:|---:|---:|---:|---:|---:|
| B | 0,716 | 0,884 | 66,0% | 91,3% | 78,7% | 0,096 |
| V4 | 0,749 | 0,892 | 75,5% | 89,6% | 82,6% | 0,087 |
| V4 head-refit | 0,765 | 0,903 | 76,9% | 89,7% | 83,3% | 0,102 |
| **Phương án khóa** | **0,798** | **0,912** | **78,6%** | **89,5%** | **84,1%** | **0,070** |
| Đối chứng không crop (B, B head-refit, cờ, điểm) | 0,801 | 0,915 | 78,7% | 89,8% | 84,3% | 0,069 |
| Chỉ detector | 0,676 | 0,833 | 62,5% | 90,7% | 76,6% | 0,092 |

| So sánh | ΔAP, CI 95% | ΔBA, CI 95% |
|---|---|---|
| Khóa − V4 | [+0,033; +0,064] | [+0,6; +2,5] điểm % |
| Khóa − V4 head-refit | [+0,020; +0,045] | [-0,1; +1,7] điểm % |
| Khóa − không crop | [-0,011; +0,004] | [-1,2; +0,8] điểm % |
| Không crop − V4 | [+0,034; +0,069] | [+0,5; +3,0] điểm % |

AP từng fold, khóa so với V4: 0,809/0,740; 0,765/0,761; 0,830/0,827; 0,766/0,705; 0,841/0,794.

QA toàn bộ ở ngưỡng 0,5: 90,73% → 90,99% (17998 câu). Đổi 167 đáp án (62 không → có, 105 có → không).

**Giới hạn:**
- Câu "Describe the finding." mâu thuẫn với đáp án gãy tăng từ 30 lên 115 trên 2457 câu, vì câu trả lời mở không đổi.
- Hậu kiểm, một seed, không có kiểm định ngoài.
- Đối chứng không crop không nằm trong 8 ứng viên; AP ngoài test của nó là 0,807 so với 0,812.
- Ứng dụng suy luận chưa được sửa; detector runtime hiện chỉ trả box, chưa trả điểm.

Tái lập: `experiments/v4_detector_scores_v1.py`, `v4_poststack_select_v1.py`, `v4_poststack_eval_v1.py`, `v4_poststack_qa_v1.py`.
