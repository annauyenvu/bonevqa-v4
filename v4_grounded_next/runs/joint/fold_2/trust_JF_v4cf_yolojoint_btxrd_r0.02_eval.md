# Lớp tin cậy của JF_v4cf_yolojoint_btxrd_r0.02 (fold 2)

Tham số khớp trên calibration (FracAtlas) và model_val (BTXRD); bảng dưới đọc trên test và external.
"Tỉ lệ trả lời" là phần ca không bị chuyển cho bác sĩ. Khoảng tin cậy 95%: bootstrap theo nhóm, 2.000 lần.

## Ngưỡng giữ 90% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,939 | 0,897 | 0,961 | +0,022 [+0,012; +0,034] | 0,429 | 40 → 23 | 0,766 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,939 | 0,914 | 0,958 | +0,019 [+0,010; +0,029] | 0,367 | 40 → 28 | 0,810 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,843 | 0,932 | 0,865 | +0,021 [+0,010; +0,034] | 0,196 | 49 → 36 | 0,652 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,843 | 0,917 | 0,862 | +0,019 [+0,007; +0,032] | 0,196 | 49 → 40 | 0,742 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,757 | 0,891 | 0,775 | +0,018 [+0,007; +0,029] | 0,173 | 106 → 92 | 0,711 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,757 | 0,849 | 0,797 | +0,040 [+0,023; +0,057] | 0,291 | 106 → 95 | 0,740 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,840 | 0,844 | 0,899 | +0,059 [+0,027; +0,096] | 0,465 | 43 → 23 | 0,856 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,840 | 0,859 | 0,896 | +0,056 [+0,026; +0,091] | 0,442 | 43 → 24 | 0,860 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,797 | 0,881 | 0,817 | +0,020 [+0,004; +0,038] | 0,208 | 106 → 84 | 0,589 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,797 | 0,868 | 0,821 | +0,024 [+0,007; +0,042] | 0,236 | 106 → 81 | 0,606 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,528 | 0,840 | 0,580 | +0,052 [+0,019; +0,087] | 0,252 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,295 | 0,916 | 0,301 | +0,006 [−0,006; +0,017] | 0,092 | — | — |

## Ngưỡng giữ 80% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,939 | 0,814 | 0,968 | +0,029 [+0,017; +0,043] | 0,571 | 40 → 17 | 0,766 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,939 | 0,822 | 0,970 | +0,031 [+0,018; +0,044] | 0,592 | 40 → 19 | 0,810 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,843 | 0,855 | 0,876 | +0,033 [+0,016; +0,052] | 0,326 | 49 → 25 | 0,652 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,843 | 0,852 | 0,886 | +0,043 [+0,025; +0,062] | 0,380 | 49 → 26 | 0,742 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,757 | 0,772 | 0,798 | +0,041 [+0,022; +0,059] | 0,357 | 106 → 72 | 0,711 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,757 | 0,738 | 0,807 | +0,050 [+0,029; +0,070] | 0,413 | 106 → 73 | 0,740 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,840 | 0,784 | 0,919 | +0,079 [+0,044; +0,116] | 0,605 | 43 → 17 | 0,856 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,840 | 0,788 | 0,925 | +0,084 [+0,047; +0,127] | 0,628 | 43 → 16 | 0,860 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,797 | 0,785 | 0,832 | +0,035 [+0,012; +0,059] | 0,349 | 106 → 69 | 0,589 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,797 | 0,774 | 0,834 | +0,037 [+0,015; +0,061] | 0,368 | 106 → 67 | 0,606 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,528 | 0,740 | 0,598 | +0,070 [+0,023; +0,122] | 0,370 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,295 | 0,835 | 0,307 | +0,012 [−0,005; +0,029] | 0,179 | — | — |

## Danh sách 3 chẩn đoán phân biệt

| Tập | Đáp án đứng đầu đúng | Đáp án đúng nằm trong 3 |
|---|---|---|
| test | 0,528 | 0,792 |
| external | 0,295 | 0,628 |
