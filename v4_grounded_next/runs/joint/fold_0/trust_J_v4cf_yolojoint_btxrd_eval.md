# Lớp tin cậy của J_v4cf_yolojoint_btxrd (fold 0)

Tham số khớp trên calibration (FracAtlas) và model_val (BTXRD); bảng dưới đọc trên test và external.
"Tỉ lệ trả lời" là phần ca không bị chuyển cho bác sĩ. Khoảng tin cậy 95%: bootstrap theo nhóm, 2.000 lần.

## Ngưỡng giữ 90% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,903 | 0,917 | 0,928 | +0,025 [+0,014; +0,037] | 0,321 | 44 → 32 | 0,771 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,903 | 0,890 | 0,958 | +0,055 [+0,039; +0,072] | 0,615 | 44 → 27 | 0,915 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,864 | 0,896 | 0,897 | +0,034 [+0,018; +0,050] | 0,325 | 37 → 30 | 0,787 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,864 | 0,913 | 0,901 | +0,037 [+0,023; +0,053] | 0,338 | 37 → 32 | 0,835 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,792 | 0,906 | 0,816 | +0,023 [+0,012; +0,036] | 0,196 | 91 → 74 | 0,634 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,792 | 0,906 | 0,820 | +0,028 [+0,015; +0,041] | 0,214 | 91 → 83 | 0,755 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,848 | 0,855 | 0,900 | +0,052 [+0,027; +0,081] | 0,439 | 27 → 19 | 0,806 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,848 | 0,885 | 0,882 | +0,035 [+0,006; +0,068] | 0,317 | 27 → 27 | 0,807 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,743 | 0,791 | 0,811 | +0,068 [+0,043; +0,094] | 0,418 | 77 → 64 | 0,687 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,743 | 0,839 | 0,817 | +0,074 [+0,046; +0,103] | 0,403 | 77 → 77 | 0,741 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,550 | 0,874 | 0,583 | +0,033 [+0,009; +0,057] | 0,190 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,287 | 0,944 | 0,288 | +0,001 [−0,009; +0,011] | 0,056 | — | — |

## Ngưỡng giữ 80% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,903 | 0,814 | 0,950 | +0,047 [+0,031; +0,063] | 0,577 | 44 → 17 | 0,771 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,903 | 0,795 | 0,973 | +0,070 [+0,052; +0,089] | 0,782 | 44 → 17 | 0,915 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,864 | 0,799 | 0,928 | +0,064 [+0,041; +0,087] | 0,575 | 37 → 21 | 0,787 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,864 | 0,831 | 0,930 | +0,067 [+0,046; +0,088] | 0,575 | 37 → 23 | 0,835 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,792 | 0,788 | 0,841 | +0,049 [+0,031; +0,068] | 0,399 | 91 → 60 | 0,634 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,792 | 0,813 | 0,837 | +0,045 [+0,028; +0,063] | 0,363 | 91 → 71 | 0,755 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,848 | 0,758 | 0,922 | +0,074 [+0,040; +0,110] | 0,610 | 27 → 16 | 0,806 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,848 | 0,758 | 0,917 | +0,069 [+0,035; +0,108] | 0,585 | 27 → 17 | 0,807 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,743 | 0,670 | 0,846 | +0,102 [+0,069; +0,136] | 0,597 | 77 → 53 | 0,687 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,743 | 0,625 | 0,850 | +0,106 [+0,072; +0,141] | 0,634 | 77 → 49 | 0,741 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,550 | 0,703 | 0,630 | +0,079 [+0,038; +0,122] | 0,421 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,287 | 0,828 | 0,301 | +0,014 [−0,004; +0,029] | 0,188 | — | — |

## Danh sách 3 chẩn đoán phân biệt

| Tập | Đáp án đứng đầu đúng | Đáp án đúng nằm trong 3 |
|---|---|---|
| test | 0,550 | 0,803 |
| external | 0,287 | 0,605 |
