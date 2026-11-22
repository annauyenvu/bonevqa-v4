# Lớp tin cậy của JF_v4cf_yolojoint_btxrd_r0.02 (fold 4)

Tham số khớp trên calibration (FracAtlas) và model_val (BTXRD); bảng dưới đọc trên test và external.
"Tỉ lệ trả lời" là phần ca không bị chuyển cho bác sĩ. Khoảng tin cậy 95%: bootstrap theo nhóm, 2.000 lần.

## Ngưỡng giữ 90% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,936 | 0,911 | 0,953 | +0,017 [+0,008; +0,027] | 0,333 | 38 → 27 | 0,738 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,936 | 0,916 | 0,955 | +0,019 [+0,009; +0,029] | 0,353 | 38 → 28 | 0,800 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,836 | 0,917 | 0,870 | +0,033 [+0,017; +0,052] | 0,271 | 59 → 36 | 0,677 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,836 | 0,918 | 0,872 | +0,036 [+0,019; +0,052] | 0,281 | 59 → 38 | 0,788 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,777 | 0,889 | 0,791 | +0,014 [+0,003; +0,026] | 0,167 | 124 → 102 | 0,695 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,777 | 0,891 | 0,799 | +0,021 [+0,009; +0,033] | 0,194 | 124 → 103 | 0,709 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,862 | 0,877 | 0,911 | +0,049 [+0,021; +0,080] | 0,432 | 28 → 21 | 0,814 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,862 | 0,888 | 0,887 | +0,025 [−0,000; +0,052] | 0,270 | 28 → 27 | 0,813 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,799 | 0,866 | 0,838 | +0,040 [+0,021; +0,061] | 0,305 | 83 → 72 | 0,662 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,799 | 0,898 | 0,827 | +0,028 [+0,012; +0,046] | 0,229 | 83 → 81 | 0,691 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,509 | 0,870 | 0,556 | +0,046 [+0,019; +0,075] | 0,212 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,284 | 0,916 | 0,297 | +0,014 [+0,002; +0,024] | 0,102 | — | — |

## Ngưỡng giữ 80% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,936 | 0,782 | 0,965 | +0,029 [+0,015; +0,041] | 0,569 | 38 → 16 | 0,738 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,936 | 0,825 | 0,965 | +0,029 [+0,016; +0,043] | 0,549 | 38 → 21 | 0,800 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,836 | 0,855 | 0,882 | +0,046 [+0,025; +0,069] | 0,385 | 59 → 27 | 0,677 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,836 | 0,857 | 0,889 | +0,052 [+0,032; +0,074] | 0,417 | 59 → 27 | 0,788 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,777 | 0,816 | 0,797 | +0,019 [+0,004; +0,037] | 0,256 | 124 → 88 | 0,695 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,777 | 0,802 | 0,801 | +0,024 [+0,009; +0,039] | 0,283 | 124 → 90 | 0,709 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,862 | 0,784 | 0,924 | +0,062 [+0,030; +0,094] | 0,568 | 28 → 16 | 0,814 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,862 | 0,755 | 0,951 | +0,088 [+0,043; +0,134] | 0,730 | 28 → 10 | 0,813 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,799 | 0,730 | 0,853 | +0,054 [+0,027; +0,080] | 0,467 | 83 → 56 | 0,662 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,799 | 0,716 | 0,850 | +0,051 [+0,027; +0,077] | 0,467 | 83 → 56 | 0,691 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,509 | 0,710 | 0,618 | +0,109 [+0,057; +0,164] | 0,447 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,284 | 0,759 | 0,301 | +0,017 [−0,007; +0,040] | 0,259 | — | — |

## Danh sách 3 chẩn đoán phân biệt

| Tập | Đáp án đứng đầu đúng | Đáp án đúng nằm trong 3 |
|---|---|---|
| test | 0,509 | 0,796 |
| external | 0,284 | 0,621 |
