# Lớp tin cậy của JF_v4cf_yolojoint_btxrd_r0.02 (fold 3)

Tham số khớp trên calibration (FracAtlas) và model_val (BTXRD); bảng dưới đọc trên test và external.
"Tỉ lệ trả lời" là phần ca không bị chuyển cho bác sĩ. Khoảng tin cậy 95%: bootstrap theo nhóm, 2.000 lần.

## Ngưỡng giữ 90% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,913 | 0,895 | 0,942 | +0,029 [+0,018; +0,041] | 0,400 | 38 → 29 | 0,740 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,913 | 0,892 | 0,951 | +0,038 [+0,025; +0,053] | 0,500 | 38 → 28 | 0,830 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,862 | 0,939 | 0,884 | +0,022 [+0,011; +0,036] | 0,210 | 43 → 32 | 0,671 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,862 | 0,920 | 0,893 | +0,031 [+0,017; +0,046] | 0,284 | 43 → 31 | 0,772 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,781 | 0,937 | 0,797 | +0,016 [+0,006; +0,026] | 0,130 | 107 → 97 | 0,674 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,781 | 0,916 | 0,800 | +0,019 [+0,008; +0,031] | 0,164 | 107 → 99 | 0,721 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,859 | 0,888 | 0,887 | +0,028 [+0,008; +0,051] | 0,289 | 34 → 27 | 0,841 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,859 | 0,874 | 0,898 | +0,039 [+0,017; +0,061] | 0,368 | 34 → 24 | 0,849 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,808 | 0,898 | 0,821 | +0,012 [−0,000; +0,025] | 0,160 | 97 → 84 | 0,593 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,808 | 0,891 | 0,824 | +0,015 [+0,003; +0,029] | 0,180 | 97 → 82 | 0,612 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,520 | 0,851 | 0,576 | +0,056 [+0,028; +0,081] | 0,248 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,280 | 0,923 | 0,290 | +0,011 [+0,001; +0,021] | 0,090 | — | — |

## Ngưỡng giữ 80% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,913 | 0,794 | 0,947 | +0,034 [+0,021; +0,049] | 0,514 | 38 → 23 | 0,740 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,913 | 0,793 | 0,956 | +0,043 [+0,029; +0,059] | 0,600 | 38 → 23 | 0,830 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,862 | 0,859 | 0,899 | +0,037 [+0,020; +0,056] | 0,370 | 43 → 22 | 0,671 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,862 | 0,835 | 0,916 | +0,054 [+0,036; +0,075] | 0,494 | 43 → 22 | 0,772 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,781 | 0,840 | 0,792 | +0,011 [−0,002; +0,026] | 0,203 | 107 → 88 | 0,674 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,781 | 0,822 | 0,812 | +0,031 [+0,016; +0,048] | 0,294 | 107 → 84 | 0,721 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,859 | 0,770 | 0,932 | +0,074 [+0,038; +0,113] | 0,632 | 34 → 14 | 0,841 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,859 | 0,755 | 0,941 | +0,082 [+0,041; +0,126] | 0,684 | 34 → 12 | 0,849 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,808 | 0,724 | 0,825 | +0,017 [−0,006; +0,040] | 0,340 | 97 → 66 | 0,593 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,808 | 0,707 | 0,837 | +0,029 [+0,005; +0,056] | 0,400 | 97 → 60 | 0,612 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,520 | 0,766 | 0,617 | +0,096 [+0,053; +0,141] | 0,388 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,280 | 0,833 | 0,301 | +0,021 [+0,006; +0,037] | 0,191 | — | — |

## Danh sách 3 chẩn đoán phân biệt

| Tập | Đáp án đứng đầu đúng | Đáp án đúng nằm trong 3 |
|---|---|---|
| test | 0,520 | 0,814 |
| external | 0,280 | 0,611 |
