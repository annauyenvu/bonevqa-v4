# B + detector huấn luyện lại + lớp nhất quán

Ngày 2026-11-01. Hậu kiểm, không thay kết quả chính của nghiên cứu V4 đã niêm phong.

**Huấn luyện:** chỉ detector, từ epoch 2 lên 6 trên TRAIN của từng fold, CPU, khoảng 2 giờ 15 phút. B và V4 không huấn luyện lại.

**Khóa trước khi mở test trong bước này** (`selection.json`, cross-fit calibration/model-validation): B + B head-refit + cờ + điểm detector cao nhất, epoch 6. Quy tắc: bộ cột đơn giản nhất, rồi epoch sớm nhất, cách tốt nhất không quá 0,005 AP. Lớp nhất quán lấy vị trí từ box detector (chọn trên cùng tập đó).

**Lưu ý:** tập ứng viên được đặt sau khi đã thấy kết quả test của bước ghép điểm trước (`../v4_poststack_v1/RESULTS.md`), nên test không hoàn toàn mù.

Số liệu: `test_results.json`, `test_qa.json`. Tái lập: `experiments/v4_detector_ext_v1.py`, `v4_detector_ext_scores_v1.py`, `v4_detector_stack_v2.py`, `v4_consistency_v1.py`, `v4_consistency_eval_v1.py`, `v4_consistency_ci_v1.py`.

## Kết quả test, 5 fold (4013 ảnh, 715 ảnh gãy, 17998 câu)

| Hệ thống | AP | AUROC | Độ nhạy | Độ đặc hiệu | BA | Brier | QA | Mâu thuẫn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 0,716 | 0,884 | 66,0% | 91,3% | 78,7% | 0,096 | 90,00% | 207 |
| V4 | 0,749 | 0,892 | 75,5% | 89,6% | 82,6% | 0,087 | 90,73% | 119 |
| V4 + detector epoch 6 (đối chứng) | 0,817 | 0,919 | 79,9% | 90,6% | 85,2% | 0,065 | | |
| **Hệ thống khóa** | **0,826** | **0,924** | **81,3%** | **90,5%** | **85,9%** | **0,063** | **91,67%** | **0** |

Hệ thống khóa − V4: ΔAP [+0,055; +0,097]; ΔBA [+2,0; +4,8] điểm %; Δđộ nhạy [+3,3; +8,4] điểm %; Δđộ đặc hiệu [-0,1; +2,1] điểm %; ΔQA [+0,55; +1,34] điểm %.
Khóa − cùng bộ chấm với detector epoch 2: ΔAP [+0,012; +0,037]. Khóa − V4 + detector: ΔAP [+0,001; +0,016].

AP từng fold, khóa so với V4: 0,834/0,740; 0,777/0,761; 0,867/0,827; 0,817/0,705; 0,859/0,794.

QA theo nhóm, khóa so với V4: Describe 83,6/80,8; abnormality 93,9/90,7; số vị trí gãy 90,0/88,3; câu gãy chính 91,6/89,9; câu gãy theo vùng 89,5/89,6; Where 48,1/50,6; câu còn lại 94,8/94,9.

**Giới hạn:** mâu thuẫn bằng 0 là do quy tắc viết lại câu trả lời; một seed; không có kiểm định ngoài; vị trí không tốt hơn (Where 48,1%, thấp hơn V4); ứng dụng suy luận chưa được sửa.

## Bổ sung 2026-11-06: V4 nhận đủ nâng cấp như hệ thống khóa (hậu kiểm)

V4 + V4 head-refit + cờ + điểm detector epoch 6 + lớp nhất quán (vị trí từ detector), cùng cross-fit, bộ chấm và bootstrap. Test đã mở từ trước. Số liệu: `v4arm_full.json`; script `experiments/v4_consistency_v4arm_v1.py`. Hệ thống khóa tái lập đúng (AP 0,826, QA 91,67%, 0 mâu thuẫn).

| Hệ thống | AP | Độ nhạy | Độ đặc hiệu | BA | Brier | QA | Mâu thuẫn |
|---|---:|---:|---:|---:|---:|---:|---:|
| V4 | 0,749 | 75,5% | 89,6% | 82,6% | 0,087 | 90,73% | 119 |
| B nâng cấp (khóa) | 0,826 | 81,3% | 90,5% | 85,9% | 0,063 | 91,67% | 0 |
| V4 nâng cấp | 0,817 | 79,9% | 90,6% | 85,2% | 0,065 | 91,95% | 0 |

V4 nâng cấp − khóa: ΔAP [−0,016; −0,001]; ΔBA [−1,7; +0,3] điểm %; ΔQA [0,00; +0,57] điểm %. V4 nâng cấp − V4: ΔAP [+0,050; +0,087]; ΔQA [+0,88; +1,58] điểm %. QA theo nhóm (V4 nâng cấp / khóa): Describe 84,3/83,6; Where 50,0/48,1; câu gãy chính 92,1/91,6.
