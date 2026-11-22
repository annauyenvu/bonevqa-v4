# Lớp tin cậy của JF_v4cf_yolojoint_btxrd_r0.02 (fold 0)

Tham số khớp trên calibration (FracAtlas) và model_val (BTXRD); bảng dưới đọc trên test và external.
"Tỉ lệ trả lời" là phần ca không bị chuyển cho bác sĩ. Khoảng tin cậy 95%: bootstrap theo nhóm, 2.000 lần.

## Ngưỡng giữ 90% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,939 | 0,914 | 0,958 | +0,019 [+0,010; +0,029] | 0,367 | 31 → 22 | 0,732 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,939 | 0,895 | 0,962 | +0,023 [+0,013; +0,035] | 0,449 | 31 → 23 | 0,820 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,860 | 0,923 | 0,886 | +0,025 [+0,012; +0,042] | 0,244 | 51 → 35 | 0,660 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,860 | 0,913 | 0,890 | +0,030 [+0,015; +0,046] | 0,280 | 51 → 38 | 0,784 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,783 | 0,845 | 0,813 | +0,029 [+0,015; +0,044] | 0,269 | 113 → 81 | 0,704 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,783 | 0,833 | 0,813 | +0,029 [+0,015; +0,044] | 0,280 | 113 → 88 | 0,730 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,885 | 0,833 | 0,946 | +0,062 [+0,029; +0,098] | 0,613 | 21 → 12 | 0,814 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,885 | 0,885 | 0,912 | +0,027 [+0,001; +0,057] | 0,323 | 21 → 21 | 0,821 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,720 | 0,782 | 0,809 | +0,089 [+0,063; +0,116] | 0,466 | 67 → 53 | 0,709 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,720 | 0,805 | 0,836 | +0,115 [+0,082; +0,149] | 0,527 | 67 → 67 | 0,785 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,543 | 0,836 | 0,578 | +0,035 [+0,003; +0,069] | 0,228 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,297 | 0,910 | 0,305 | +0,008 [−0,003; +0,020] | 0,101 | — | — |

## Ngưỡng giữ 80% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,939 | 0,841 | 0,964 | +0,025 [+0,014; +0,039] | 0,510 | 31 → 16 | 0,732 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,939 | 0,822 | 0,968 | +0,029 [+0,017; +0,042] | 0,571 | 31 → 18 | 0,820 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,860 | 0,852 | 0,896 | +0,036 [+0,017; +0,056] | 0,366 | 51 → 25 | 0,660 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,860 | 0,835 | 0,902 | +0,042 [+0,023; +0,062] | 0,415 | 51 → 28 | 0,784 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,783 | 0,761 | 0,826 | +0,043 [+0,023; +0,064] | 0,389 | 113 → 60 | 0,704 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,783 | 0,736 | 0,835 | +0,052 [+0,032; +0,073] | 0,440 | 113 → 62 | 0,730 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,885 | 0,740 | 0,955 | +0,070 [+0,034; +0,107] | 0,710 | 21 → 9 | 0,814 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,885 | 0,743 | 0,940 | +0,055 [+0,019; +0,093] | 0,613 | 21 → 12 | 0,821 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,720 | 0,632 | 0,855 | +0,134 [+0,099; +0,171] | 0,671 | 67 → 47 | 0,709 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,720 | 0,642 | 0,848 | +0,127 [+0,091; +0,163] | 0,651 | 67 → 51 | 0,785 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,543 | 0,736 | 0,636 | +0,094 [+0,052; +0,140] | 0,415 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,297 | 0,768 | 0,312 | +0,015 [−0,008; +0,037] | 0,248 | — | — |

## Danh sách 3 chẩn đoán phân biệt

| Tập | Đáp án đứng đầu đúng | Đáp án đúng nằm trong 3 |
|---|---|---|
| test | 0,543 | 0,799 |
| external | 0,297 | 0,603 |
