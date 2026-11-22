# Lớp tin cậy của J_v4cf_yolojoint_btxrd (fold 4)

Tham số khớp trên calibration (FracAtlas) và model_val (BTXRD); bảng dưới đọc trên test và external.
"Tỉ lệ trả lời" là phần ca không bị chuyển cho bác sĩ. Khoảng tin cậy 95%: bootstrap theo nhóm, 2.000 lần.

## Ngưỡng giữ 90% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,901 | 0,945 | 0,922 | +0,021 [+0,010; +0,032] | 0,253 | 45 → 35 | 0,719 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,901 | 0,919 | 0,943 | +0,042 [+0,028; +0,057] | 0,468 | 45 → 29 | 0,881 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,843 | 0,865 | 0,894 | +0,050 [+0,031; +0,072] | 0,413 | 50 → 32 | 0,751 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,843 | 0,896 | 0,897 | +0,054 [+0,035; +0,075] | 0,413 | 50 → 33 | 0,821 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,767 | 0,845 | 0,795 | +0,028 [+0,013; +0,043] | 0,255 | 109 → 85 | 0,642 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,767 | 0,892 | 0,803 | +0,036 [+0,022; +0,052] | 0,245 | 109 → 87 | 0,764 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,851 | 0,851 | 0,908 | +0,057 [+0,028; +0,089] | 0,475 | 24 → 20 | 0,795 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,851 | 0,862 | 0,897 | +0,045 [+0,013; +0,081] | 0,400 | 24 → 24 | 0,794 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,795 | 0,822 | 0,841 | +0,046 [+0,027; +0,067] | 0,364 | 74 → 65 | 0,621 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,795 | 0,889 | 0,838 | +0,043 [+0,024; +0,065] | 0,299 | 74 → 74 | 0,652 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,550 | 0,874 | 0,587 | +0,037 [+0,013; +0,064] | 0,198 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,289 | 0,914 | 0,300 | +0,011 [−0,001; +0,021] | 0,100 | — | — |

## Ngưỡng giữ 80% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,901 | 0,848 | 0,941 | +0,040 [+0,025; +0,055] | 0,494 | 45 → 21 | 0,719 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,901 | 0,834 | 0,966 | +0,064 [+0,047; +0,082] | 0,709 | 45 → 21 | 0,881 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,843 | 0,784 | 0,904 | +0,061 [+0,039; +0,085] | 0,522 | 50 → 29 | 0,751 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,843 | 0,796 | 0,919 | +0,075 [+0,051; +0,101] | 0,587 | 50 → 27 | 0,821 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,767 | 0,761 | 0,807 | +0,039 [+0,019; +0,060] | 0,367 | 109 → 74 | 0,642 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,767 | 0,788 | 0,829 | +0,062 [+0,042; +0,081] | 0,420 | 109 → 72 | 0,764 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,851 | 0,758 | 0,922 | +0,070 [+0,033; +0,108] | 0,600 | 24 → 16 | 0,795 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,851 | 0,740 | 0,935 | +0,083 [+0,042; +0,126] | 0,675 | 24 → 13 | 0,794 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,795 | 0,715 | 0,845 | +0,049 [+0,026; +0,073] | 0,458 | 74 → 58 | 0,621 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,795 | 0,716 | 0,842 | +0,047 [+0,023; +0,071] | 0,449 | 74 → 59 | 0,652 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,550 | 0,758 | 0,618 | +0,067 [+0,031; +0,109] | 0,355 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,289 | 0,818 | 0,309 | +0,020 [−0,001; +0,040] | 0,205 | — | — |

## Danh sách 3 chẩn đoán phân biệt

| Tập | Đáp án đứng đầu đúng | Đáp án đúng nằm trong 3 |
|---|---|---|
| test | 0,550 | 0,788 |
| external | 0,289 | 0,626 |
