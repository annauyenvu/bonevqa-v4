# Lớp tin cậy của J_v4cf_yolojoint_btxrd (fold 1)

Tham số khớp trên calibration (FracAtlas) và model_val (BTXRD); bảng dưới đọc trên test và external.
"Tỉ lệ trả lời" là phần ca không bị chuyển cho bác sĩ. Khoảng tin cậy 95%: bootstrap theo nhóm, 2.000 lần.

## Ngưỡng giữ 90% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,905 | 0,889 | 0,941 | +0,036 [+0,023; +0,051] | 0,447 | 39 → 20 | 0,729 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,905 | 0,888 | 0,948 | +0,043 [+0,029; +0,057] | 0,513 | 39 → 26 | 0,844 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,838 | 0,911 | 0,875 | +0,037 [+0,021; +0,055] | 0,295 | 76 → 56 | 0,774 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,838 | 0,915 | 0,875 | +0,037 [+0,021; +0,056] | 0,295 | 76 → 56 | 0,779 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,739 | 0,899 | 0,766 | +0,027 [+0,015; +0,039] | 0,194 | 174 → 150 | 0,722 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,739 | 0,899 | 0,766 | +0,027 [+0,015; +0,039] | 0,194 | 174 → 150 | 0,731 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,851 | 0,892 | 0,887 | +0,036 [+0,014; +0,061] | 0,325 | 36 → 27 | 0,781 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,851 | 0,918 | 0,887 | +0,035 [+0,011; +0,064] | 0,300 | 36 → 26 | 0,815 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,807 | 0,839 | 0,829 | +0,022 [+0,003; +0,042] | 0,257 | 91 → 75 | 0,538 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,807 | 0,895 | 0,839 | +0,033 [+0,017; +0,050] | 0,257 | 91 → 75 | 0,666 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,532 | 0,896 | 0,560 | +0,029 [+0,007; +0,054] | 0,159 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,284 | 0,929 | 0,287 | +0,003 [−0,009; +0,014] | 0,075 | — | — |

## Ngưỡng giữ 80% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,905 | 0,802 | 0,944 | +0,039 [+0,024; +0,054] | 0,526 | 39 → 16 | 0,729 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,905 | 0,796 | 0,962 | +0,057 [+0,042; +0,074] | 0,684 | 39 → 17 | 0,844 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,838 | 0,836 | 0,892 | +0,054 [+0,033; +0,076] | 0,442 | 76 → 43 | 0,774 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,838 | 0,840 | 0,892 | +0,054 [+0,035; +0,077] | 0,442 | 76 → 43 | 0,779 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,739 | 0,803 | 0,781 | +0,042 [+0,025; +0,059] | 0,327 | 174 → 124 | 0,722 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,739 | 0,809 | 0,780 | +0,041 [+0,024; +0,058] | 0,318 | 174 → 125 | 0,731 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,851 | 0,710 | 0,927 | +0,075 [+0,038; +0,118] | 0,650 | 36 → 14 | 0,781 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,851 | 0,747 | 0,930 | +0,079 [+0,039; +0,121] | 0,650 | 36 → 14 | 0,815 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,807 | 0,711 | 0,825 | +0,018 [−0,006; +0,042] | 0,356 | 91 → 65 | 0,538 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,807 | 0,720 | 0,843 | +0,037 [+0,013; +0,060] | 0,416 | 91 → 59 | 0,666 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,532 | 0,773 | 0,582 | +0,050 [+0,013; +0,096] | 0,310 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,284 | 0,831 | 0,283 | −0,000 [−0,020; +0,018] | 0,168 | — | — |

## Danh sách 3 chẩn đoán phân biệt

| Tập | Đáp án đứng đầu đúng | Đáp án đúng nằm trong 3 |
|---|---|---|
| test | 0,532 | 0,792 |
| external | 0,284 | 0,625 |
