# Lớp tin cậy của J_v4cf_yolojoint_btxrd (fold 3)

Tham số khớp trên calibration (FracAtlas) và model_val (BTXRD); bảng dưới đọc trên test và external.
"Tỉ lệ trả lời" là phần ca không bị chuyển cho bác sĩ. Khoảng tin cậy 95%: bootstrap theo nhóm, 2.000 lần.

## Ngưỡng giữ 90% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,901 | 0,899 | 0,922 | +0,021 [+0,010; +0,032] | 0,291 | 60 → 45 | 0,719 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,901 | 0,892 | 0,948 | +0,047 [+0,033; +0,062] | 0,532 | 60 → 28 | 0,842 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,821 | 0,908 | 0,854 | +0,033 [+0,018; +0,049] | 0,257 | 52 → 42 | 0,756 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,821 | 0,928 | 0,870 | +0,049 [+0,031; +0,067] | 0,324 | 52 → 38 | 0,874 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,717 | 0,892 | 0,736 | +0,020 [+0,007; +0,033] | 0,170 | 113 → 98 | 0,637 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,717 | 0,908 | 0,763 | +0,046 [+0,031; +0,062] | 0,240 | 113 → 89 | 0,815 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,851 | 0,851 | 0,908 | +0,057 [+0,030; +0,088] | 0,475 | 26 → 21 | 0,788 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,851 | 0,866 | 0,893 | +0,041 [+0,015; +0,070] | 0,375 | 26 → 25 | 0,786 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,741 | 0,870 | 0,782 | +0,041 [+0,023; +0,060] | 0,267 | 83 → 77 | 0,656 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,741 | 0,843 | 0,814 | +0,072 [+0,047; +0,100] | 0,393 | 83 → 82 | 0,708 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,532 | 0,885 | 0,571 | +0,040 [+0,015; +0,069] | 0,190 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,272 | 0,937 | 0,280 | +0,008 [−0,003; +0,018] | 0,074 | — | — |

## Ngưỡng giữ 80% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,901 | 0,815 | 0,937 | +0,036 [+0,022; +0,051] | 0,481 | 60 → 31 | 0,719 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,901 | 0,813 | 0,954 | +0,052 [+0,036; +0,070] | 0,620 | 60 → 21 | 0,842 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,821 | 0,836 | 0,870 | +0,049 [+0,028; +0,070] | 0,390 | 52 → 37 | 0,756 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,821 | 0,848 | 0,906 | +0,084 [+0,063; +0,109] | 0,552 | 52 → 29 | 0,874 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,717 | 0,800 | 0,757 | +0,040 [+0,022; +0,058] | 0,314 | 113 → 86 | 0,637 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,717 | 0,806 | 0,806 | +0,090 [+0,069; +0,112] | 0,450 | 113 → 72 | 0,815 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,851 | 0,732 | 0,919 | +0,067 [+0,031; +0,108] | 0,600 | 26 → 16 | 0,788 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,851 | 0,755 | 0,921 | +0,070 [+0,035; +0,109] | 0,600 | 26 → 16 | 0,786 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,741 | 0,734 | 0,809 | +0,068 [+0,040; +0,095] | 0,459 | 83 → 73 | 0,656 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,741 | 0,739 | 0,819 | +0,077 [+0,050; +0,106] | 0,481 | 83 → 70 | 0,708 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,532 | 0,770 | 0,599 | +0,067 [+0,032; +0,103] | 0,341 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,272 | 0,828 | 0,296 | +0,024 [+0,008; +0,040] | 0,200 | — | — |

## Danh sách 3 chẩn đoán phân biệt

| Tập | Đáp án đứng đầu đúng | Đáp án đúng nằm trong 3 |
|---|---|---|
| test | 0,532 | 0,810 |
| external | 0,272 | 0,630 |
