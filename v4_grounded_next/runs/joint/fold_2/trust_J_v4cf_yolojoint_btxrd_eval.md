# Lớp tin cậy của J_v4cf_yolojoint_btxrd (fold 2)

Tham số khớp trên calibration (FracAtlas) và model_val (BTXRD); bảng dưới đọc trên test và external.
"Tỉ lệ trả lời" là phần ca không bị chuyển cho bác sĩ. Khoảng tin cậy 95%: bootstrap theo nhóm, 2.000 lần.

## Ngưỡng giữ 90% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,913 | 0,893 | 0,937 | +0,024 [+0,012; +0,039] | 0,357 | 40 → 30 | 0,784 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,913 | 0,898 | 0,951 | +0,039 [+0,024; +0,054] | 0,500 | 40 → 33 | 0,838 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,842 | 0,915 | 0,864 | +0,022 [+0,009; +0,037] | 0,215 | 53 → 41 | 0,724 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,842 | 0,911 | 0,865 | +0,024 [+0,010; +0,039] | 0,226 | 53 → 43 | 0,777 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,764 | 0,873 | 0,789 | +0,025 [+0,010; +0,041] | 0,220 | 111 → 93 | 0,636 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,764 | 0,855 | 0,787 | +0,024 [+0,008; +0,040] | 0,230 | 111 → 97 | 0,735 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,866 | 0,888 | 0,904 | +0,038 [+0,015; +0,064] | 0,361 | 31 → 22 | 0,768 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,866 | 0,918 | 0,887 | +0,020 [+0,001; +0,040] | 0,222 | 31 → 28 | 0,747 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,768 | 0,858 | 0,812 | +0,044 [+0,027; +0,064] | 0,306 | 81 → 70 | 0,632 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,768 | 0,851 | 0,831 | +0,063 [+0,041; +0,088] | 0,380 | 81 → 75 | 0,700 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,532 | 0,866 | 0,558 | +0,026 [+0,001; +0,050] | 0,183 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,289 | 0,912 | 0,300 | +0,011 [−0,000; +0,023] | 0,102 | — | — |

## Ngưỡng giữ 80% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,913 | 0,822 | 0,950 | +0,037 [+0,021; +0,055] | 0,529 | 40 → 23 | 0,784 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,913 | 0,803 | 0,958 | +0,045 [+0,029; +0,063] | 0,614 | 40 → 26 | 0,838 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,842 | 0,842 | 0,879 | +0,037 [+0,019; +0,057] | 0,355 | 53 → 35 | 0,724 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,842 | 0,811 | 0,895 | +0,053 [+0,033; +0,076] | 0,462 | 53 → 37 | 0,777 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,764 | 0,785 | 0,801 | +0,038 [+0,019; +0,058] | 0,340 | 111 → 78 | 0,636 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,764 | 0,774 | 0,814 | +0,051 [+0,031; +0,072] | 0,393 | 111 → 81 | 0,735 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,866 | 0,762 | 0,917 | +0,051 [+0,023; +0,082] | 0,528 | 31 → 17 | 0,768 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,866 | 0,777 | 0,914 | +0,048 [+0,022; +0,074] | 0,500 | 31 → 18 | 0,747 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,768 | 0,716 | 0,826 | +0,058 [+0,031; +0,085] | 0,463 | 81 → 64 | 0,632 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,768 | 0,711 | 0,830 | +0,062 [+0,035; +0,091] | 0,479 | 81 → 63 | 0,700 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,532 | 0,732 | 0,604 | +0,072 [+0,033; +0,112] | 0,381 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,289 | 0,778 | 0,313 | +0,024 [+0,001; +0,046] | 0,248 | — | — |

## Danh sách 3 chẩn đoán phân biệt

| Tập | Đáp án đứng đầu đúng | Đáp án đúng nằm trong 3 |
|---|---|---|
| test | 0,532 | 0,807 |
| external | 0,289 | 0,638 |
