# Lớp tin cậy của J_s43_v4cf_yolojoint_btxrd (fold 0)

Tham số khớp trên calibration (FracAtlas) và model_val (BTXRD); bảng dưới đọc trên test và external.
"Tỉ lệ trả lời" là phần ca không bị chuyển cho bác sĩ. Khoảng tin cậy 95%: bootstrap theo nhóm, 2.000 lần.

## Ngưỡng giữ 90% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,897 | 0,897 | 0,936 | +0,039 [+0,026; +0,053] | 0,446 | 44 → 25 | 0,782 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,897 | 0,893 | 0,964 | +0,067 [+0,050; +0,085] | 0,687 | 44 → 23 | 0,925 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,867 | 0,932 | 0,887 | +0,020 [+0,008; +0,031] | 0,205 | 41 → 33 | 0,751 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,867 | 0,922 | 0,902 | +0,035 [+0,020; +0,053] | 0,321 | 41 → 35 | 0,822 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,782 | 0,928 | 0,805 | +0,023 [+0,012; +0,035] | 0,170 | 98 → 82 | 0,607 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,782 | 0,895 | 0,819 | +0,037 [+0,023; +0,050] | 0,256 | 98 → 87 | 0,758 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,848 | 0,874 | 0,881 | +0,033 [+0,011; +0,059] | 0,317 | 39 → 28 | 0,746 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,848 | 0,896 | 0,888 | +0,040 [+0,015; +0,069] | 0,341 | 39 → 25 | 0,751 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,812 | 0,883 | 0,818 | +0,006 [−0,009; +0,021] | 0,143 | 93 → 84 | 0,528 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,812 | 0,889 | 0,828 | +0,015 [−0,000; +0,034] | 0,184 | 93 → 76 | 0,577 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,554 | 0,877 | 0,585 | +0,031 [+0,007; +0,057] | 0,183 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,282 | 0,927 | 0,285 | +0,004 [−0,009; +0,015] | 0,077 | — | — |

## Ngưỡng giữ 80% số ca trên val

| Câu hỏi | Tập | Tín hiệu | Đúng (tất cả) | Tỉ lệ trả lời | Đúng (đã trả lời) | Chênh [KTC 95%] | Lỗi được chuyển | Bỏ sót dương tính | AUROC đúng / sai |
|---|---|---|---|---|---|---|---|---|---|
| Gãy xương (FracAtlas) | test | chỉ đầu ra có / không | 0,897 | 0,829 | 0,947 | +0,051 [+0,035; +0,066] | 0,578 | 44 → 17 | 0,782 |
| Gãy xương (FracAtlas) | test | đầu ra + điểm detector | 0,897 | 0,809 | 0,972 | +0,076 [+0,058; +0,094] | 0,783 | 44 → 17 | 0,925 |
| Có u (BTXRD) | test | chỉ đầu ra có / không | 0,867 | 0,838 | 0,913 | +0,045 [+0,028; +0,065] | 0,449 | 41 → 26 | 0,751 |
| Có u (BTXRD) | test | đầu ra + điểm detector | 0,867 | 0,847 | 0,920 | +0,052 [+0,033; +0,071] | 0,487 | 41 → 27 | 0,822 |
| Có u (BTXRD) | external | chỉ đầu ra có / không | 0,782 | 0,838 | 0,826 | +0,044 [+0,028; +0,062] | 0,330 | 98 → 73 | 0,607 |
| Có u (BTXRD) | external | đầu ra + điểm detector | 0,782 | 0,832 | 0,833 | +0,051 [+0,035; +0,068] | 0,364 | 98 → 78 | 0,758 |
| Lành / ác (BTXRD) | test | chỉ đầu ra có / không | 0,848 | 0,736 | 0,894 | +0,046 [+0,015; +0,083] | 0,488 | 39 → 21 | 0,746 |
| Lành / ác (BTXRD) | test | đầu ra + điểm detector | 0,848 | 0,784 | 0,891 | +0,043 [+0,011; +0,078] | 0,439 | 39 → 21 | 0,751 |
| Lành / ác (BTXRD) | external | chỉ đầu ra có / không | 0,812 | 0,716 | 0,826 | +0,014 [−0,010; +0,037] | 0,337 | 93 → 65 | 0,528 |
| Lành / ác (BTXRD) | external | đầu ra + điểm detector | 0,812 | 0,759 | 0,836 | +0,024 [−0,001; +0,047] | 0,337 | 93 → 61 | 0,577 |
| Chẩn đoán loại u (BTXRD) | test | điểm 9 đáp án ứng viên | 0,554 | 0,699 | 0,628 | +0,074 [+0,030; +0,127] | 0,417 | — | — |
| Chẩn đoán loại u (BTXRD) | external | điểm 9 đáp án ứng viên | 0,282 | 0,759 | 0,295 | +0,014 [−0,010; +0,039] | 0,256 | — | — |

## Danh sách 3 chẩn đoán phân biệt

| Tập | Đáp án đứng đầu đúng | Đáp án đúng nằm trong 3 |
|---|---|---|
| test | 0,554 | 0,799 |
| external | 0,282 | 0,613 |
