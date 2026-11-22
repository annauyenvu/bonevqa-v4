# Lớp tin cậy của J_s44_v4cf_yolojoint_btxrd (fold 0)

Tham số khớp trên calibration (FracAtlas) và model_val (BTXRD); bảng dưới đọc trên test và external.
"Tỉ lệ trả lời" là phần ca không bị chuyển cho bác sĩ. Khoảng tin cậy 95%: bootstrap theo nhóm, 2.000 lần.

## Ngưỡng giữ 90% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,902 | 0,887 | 0,933 | +0,031 [+0,018; +0,044] | 0,392 | 41 → 25 | 0,772 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,902 | 0,899 | 0,957 | +0,055 [+0,040; +0,072] | 0,608 | 41 → 27 | 0,915 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,865 | 0,913 | 0,890 | +0,025 [+0,012; +0,038] | 0,253 | 41 → 34 | 0,780 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,865 | 0,896 | 0,901 | +0,036 [+0,021; +0,053] | 0,342 | 41 → 35 | 0,839 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,796 | 0,922 | 0,808 | +0,012 [+0,003; +0,022] | 0,133 | 94 → 81 | 0,594 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,796 | 0,881 | 0,822 | +0,026 [+0,013; +0,039] | 0,230 | 94 → 79 | 0,733 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,844 | 0,903 | 0,877 | +0,033 [+0,011; +0,056] | 0,286 | 42 → 30 | 0,809 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,844 | 0,877 | 0,890 | +0,046 [+0,020; +0,075] | 0,381 | 42 → 26 | 0,824 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,797 | 0,874 | 0,809 | +0,012 [−0,004; +0,030] | 0,179 | 106 → 87 | 0,568 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,797 | 0,845 | 0,825 | +0,028 [+0,009; +0,050] | 0,274 | 106 → 77 | 0,610 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,520 | 0,844 | 0,546 | +0,026 [−0,005; +0,055] | 0,202 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,284 | 0,920 | 0,296 | +0,012 [+0,002; +0,023] | 0,096 | — | — |

## Ngưỡng giữ 80% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,902 | 0,785 | 0,951 | +0,049 [+0,033; +0,067] | 0,608 | 41 → 14 | 0,772 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,902 | 0,798 | 0,977 | +0,075 [+0,057; +0,094] | 0,810 | 41 → 15 | 0,915 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,865 | 0,816 | 0,923 | +0,057 [+0,037; +0,078] | 0,532 | 41 → 25 | 0,780 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,865 | 0,821 | 0,934 | +0,068 [+0,047; +0,091] | 0,595 | 41 → 20 | 0,839 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,796 | 0,775 | 0,845 | +0,049 [+0,032; +0,067] | 0,412 | 94 → 64 | 0,594 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,796 | 0,772 | 0,843 | +0,047 [+0,030; +0,066] | 0,406 | 94 → 60 | 0,733 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,844 | 0,729 | 0,923 | +0,080 [+0,038; +0,125] | 0,643 | 42 → 15 | 0,809 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,844 | 0,725 | 0,923 | +0,079 [+0,037; +0,126] | 0,643 | 42 → 15 | 0,824 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,797 | 0,684 | 0,829 | +0,032 [+0,003; +0,063] | 0,425 | 106 → 61 | 0,568 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,797 | 0,661 | 0,832 | +0,035 [+0,003; +0,068] | 0,453 | 106 → 58 | 0,610 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,520 | 0,665 | 0,609 | +0,088 [+0,041; +0,146] | 0,457 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,284 | 0,734 | 0,311 | +0,027 [+0,003; +0,050] | 0,294 | — | — |

## Danh sách 3 chẩn đoán phân biệt

| Tập | Đáp án đứng đầu đúng | Đáp án đúng nằm trong 3 |
|---|---|---|
| test | 0,520 | 0,788 |
| external | 0,284 | 0,617 |
