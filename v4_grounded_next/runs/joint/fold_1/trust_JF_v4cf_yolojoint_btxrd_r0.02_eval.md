# Lớp tin cậy của JF_v4cf_yolojoint_btxrd_r0.02 (fold 1)

Tham số khớp trên calibration (FracAtlas) và model_val (BTXRD); bảng dưới đọc trên test và external.
"Tỉ lệ trả lời" là phần ca không bị chuyển cho bác sĩ. Khoảng tin cậy 95%: bootstrap theo nhóm, 2.000 lần.

## Ngưỡng giữ 90% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,917 | 0,899 | 0,943 | +0,027 [+0,015; +0,039] | 0,388 | 39 → 25 | 0,721 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,917 | 0,910 | 0,947 | +0,030 [+0,019; +0,043] | 0,418 | 39 → 30 | 0,834 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,855 | 0,928 | 0,879 | +0,024 [+0,011; +0,039] | 0,224 | 42 → 28 | 0,643 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,855 | 0,886 | 0,890 | +0,035 [+0,019; +0,053] | 0,329 | 42 → 34 | 0,776 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,782 | 0,920 | 0,798 | +0,016 [+0,007; +0,026] | 0,148 | 113 → 98 | 0,677 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,782 | 0,871 | 0,811 | +0,029 [+0,013; +0,046] | 0,244 | 113 → 104 | 0,680 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,866 | 0,885 | 0,887 | +0,020 [+0,002; +0,041] | 0,250 | 33 → 27 | 0,792 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,866 | 0,907 | 0,885 | +0,019 [−0,000; +0,044] | 0,222 | 33 → 26 | 0,769 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,805 | 0,866 | 0,830 | +0,025 [+0,008; +0,043] | 0,245 | 90 → 76 | 0,598 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,805 | 0,870 | 0,835 | +0,030 [+0,013; +0,050] | 0,265 | 90 → 75 | 0,686 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,506 | 0,825 | 0,545 | +0,039 [+0,014; +0,068] | 0,241 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,278 | 0,837 | 0,279 | +0,001 [−0,017; +0,021] | 0,164 | — | — |

## Ngưỡng giữ 80% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,917 | 0,793 | 0,954 | +0,038 [+0,023; +0,053] | 0,567 | 39 → 14 | 0,721 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,917 | 0,782 | 0,965 | +0,048 [+0,033; +0,064] | 0,672 | 39 → 16 | 0,834 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,855 | 0,862 | 0,885 | +0,030 [+0,015; +0,047] | 0,318 | 42 → 22 | 0,643 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,855 | 0,780 | 0,926 | +0,071 [+0,048; +0,095] | 0,600 | 42 → 25 | 0,776 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,782 | 0,839 | 0,805 | +0,023 [+0,010; +0,037] | 0,250 | 113 → 83 | 0,677 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,782 | 0,722 | 0,813 | +0,031 [+0,011; +0,054] | 0,381 | 113 → 94 | 0,680 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,866 | 0,725 | 0,933 | +0,067 [+0,032; +0,107] | 0,639 | 33 → 13 | 0,792 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,866 | 0,796 | 0,930 | +0,064 [+0,030; +0,101] | 0,583 | 33 → 15 | 0,769 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,805 | 0,738 | 0,839 | +0,034 [+0,011; +0,057] | 0,392 | 90 → 62 | 0,598 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,805 | 0,778 | 0,850 | +0,045 [+0,023; +0,069] | 0,402 | 90 → 61 | 0,686 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,506 | 0,725 | 0,574 | +0,069 [+0,027; +0,112] | 0,376 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,278 | 0,711 | 0,299 | +0,021 [−0,005; +0,047] | 0,310 | — | — |

## Danh sách 3 chẩn đoán phân biệt

| Tập | Đáp án đứng đầu đúng | Đáp án đúng nằm trong 3 |
|---|---|---|
| test | 0,506 | 0,788 |
| external | 0,278 | 0,625 |
