# Bảng tổng hợp cấu hình cuối cùng (JF), fold 0–4

Tạo lúc 2026-11-22_02:55:13 từ các file JSON đã lưu. Mỗi fold một seed (42). Dấu * là CI 95% không chứa 0.

**Các nhánh:** A = V4 gốc niêm phong (Faster R-CNN). B = V4-CF huấn luyện lại, Faster R-CNN. D2 = V4-CF + YOLOv8 joint (ngưỡng theo tỉ lệ của Faster R-CNN). J = D2 + câu hỏi–đáp BTXRD. **JF = cấu hình cuối cùng**: J với ngưỡng riêng của YOLOv8 (FracAtlas mức 0,02, chọn trên model_val fold 0; BTXRD chọn trên model_val fold 0 chỉ ở bước suy luận, xem bảng 4e). A, B, D2, J chỉ là tham chiếu.

## 0. Detector YOLOv8 riêng (không qua V4)

Detector chính (joint FracAtlas + BTXRD, YOLOv8s) của mỗi fold, **không huấn luyện lại, không bị thay đổi** bởi bất kỳ nhánh VQA nào. Điểm vận hành riêng của detector = ngưỡng có F1 hộp @IoU 0,5 cao nhất trên model_val (chọn theo từng fold, từng lớp). Đây là ngưỡng dùng để **hiển thị hộp** cho người đọc phim. Cột "mức đưa vào V4" là ngưỡng proposal của JF (FracAtlas 0,02; BTXRD mức chọn ở bảng 4e), chỉ để V4 dùng.

| Fold | Tập | Ngưỡng | Precision hộp @0,5 | Recall hộp @0,5 | F1 @0,5 | Ảnh bệnh có hộp trúng (@0,3) | Ảnh bình thường có hộp |
|---|---|---|---|---|---|---|---|
| 0 | FracAtlas gãy test, riêng của detector | 0.110 | 0.590 | 0.520 | 0.552 | 0.685 | 0.030 |
| 1 | FracAtlas gãy test, riêng của detector | 0.084 | 0.492 | 0.526 | 0.508 | 0.643 | 0.089 |
| 2 | FracAtlas gãy test, riêng của detector | 0.546 | 0.786 | 0.349 | 0.484 | 0.427 | 0.003 |
| 3 | FracAtlas gãy test, riêng của detector | 0.387 | 0.707 | 0.489 | 0.578 | 0.580 | 0.018 |
| 4 | FracAtlas gãy test, riêng của detector | 0.547 | 0.635 | 0.346 | 0.448 | 0.483 | 0.011 |
| **TB** | FracAtlas gãy test, riêng của detector | | 0.642 | 0.446 | 0.514 | 0.564 | 0.030 |
| 0 | FracAtlas gãy test, mức đưa vào V4 | 0.226 | 0.655 | 0.492 | 0.562 | 0.657 | 0.015 |
| 1 | FracAtlas gãy test, mức đưa vào V4 | 0.426 | 0.774 | 0.453 | 0.571 | 0.524 | 0.020 |
| 2 | FracAtlas gãy test, mức đưa vào V4 | 0.372 | 0.725 | 0.413 | 0.526 | 0.497 | 0.011 |
| 3 | FracAtlas gãy test, mức đưa vào V4 | 0.270 | 0.669 | 0.533 | 0.593 | 0.629 | 0.021 |
| 4 | FracAtlas gãy test, mức đưa vào V4 | 0.440 | 0.636 | 0.390 | 0.483 | 0.524 | 0.014 |
| **TB** | FracAtlas gãy test, mức đưa vào V4 | | 0.692 | 0.456 | 0.547 | 0.566 | 0.016 |
| 0 | BTXRD u test, riêng của detector | 0.126 | 0.567 | 0.575 | 0.571 | 0.732 | 0.088 |
| 1 | BTXRD u test, riêng của detector | 0.261 | 0.639 | 0.466 | 0.539 | 0.613 | 0.044 |
| 2 | BTXRD u test, riêng của detector | 0.230 | 0.537 | 0.463 | 0.497 | 0.613 | 0.066 |
| 3 | BTXRD u test, riêng của detector | 0.229 | 0.590 | 0.514 | 0.550 | 0.688 | 0.050 |
| 4 | BTXRD u test, riêng của detector | 0.106 | 0.488 | 0.540 | 0.513 | 0.732 | 0.119 |
| **TB** | BTXRD u test, riêng của detector | | 0.564 | 0.511 | 0.534 | 0.676 | 0.074 |
| 0 | BTXRD u test, mức đưa vào V4 | 0.122 | 0.566 | 0.575 | 0.570 | 0.732 | 0.088 |
| 1 | BTXRD u test, mức đưa vào V4 | 0.131 | 0.541 | 0.537 | 0.539 | 0.706 | 0.107 |
| 2 | BTXRD u test, mức đưa vào V4 | 0.168 | 0.483 | 0.494 | 0.488 | 0.673 | 0.110 |
| 3 | BTXRD u test, mức đưa vào V4 | 0.116 | 0.528 | 0.557 | 0.542 | 0.740 | 0.104 |
| 4 | BTXRD u test, mức đưa vào V4 | 0.124 | 0.509 | 0.534 | 0.522 | 0.710 | 0.107 |
| **TB** | BTXRD u test, mức đưa vào V4 | | 0.525 | 0.540 | 0.532 | 0.712 | 0.103 |
| 0 | BTXRD u external, riêng của detector | 0.126 | 0.555 | 0.491 | 0.521 | 0.665 | 0.157 |
| 1 | BTXRD u external, riêng của detector | 0.261 | 0.601 | 0.381 | 0.466 | 0.536 | 0.070 |
| 2 | BTXRD u external, riêng của detector | 0.230 | 0.598 | 0.435 | 0.504 | 0.603 | 0.087 |
| 3 | BTXRD u external, riêng của detector | 0.229 | 0.600 | 0.477 | 0.531 | 0.655 | 0.112 |
| 4 | BTXRD u external, riêng của detector | 0.106 | 0.508 | 0.491 | 0.499 | 0.686 | 0.185 |
| **TB** | BTXRD u external, riêng của detector | | 0.572 | 0.455 | 0.504 | 0.629 | 0.122 |
| 0 | BTXRD u external, mức đưa vào V4 | 0.122 | 0.552 | 0.491 | 0.519 | 0.665 | 0.157 |
| 1 | BTXRD u external, mức đưa vào V4 | 0.131 | 0.539 | 0.456 | 0.494 | 0.642 | 0.150 |
| 2 | BTXRD u external, mức đưa vào V4 | 0.168 | 0.552 | 0.464 | 0.504 | 0.640 | 0.154 |
| 3 | BTXRD u external, mức đưa vào V4 | 0.116 | 0.531 | 0.518 | 0.524 | 0.713 | 0.210 |
| 4 | BTXRD u external, mức đưa vào V4 | 0.124 | 0.523 | 0.480 | 0.501 | 0.672 | 0.175 |
| **TB** | BTXRD u external, mức đưa vào V4 | | 0.540 | 0.482 | 0.509 | 0.666 | 0.169 |

Nguyên tắc: ngưỡng đưa vào V4 chỉ quyết định V4 nhìn thấy hộp nào; nó không thay đổi detector, và hộp hiển thị cho người dùng dùng điểm vận hành riêng của detector. Không tối ưu một nhánh VQA trung gian bằng cách làm giảm hiệu năng detector.

## 1. FracAtlas test (phát hiện gãy xương)

| Fold | A | B | D2 | J | **JF** | JF − A | JF − B | JF − J |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.740 | 0.732 | 0.784 | 0.799 | 0.863 | +0.123 [+0.076; +0.174] * | +0.131 [+0.084; +0.182] * | +0.065 [+0.026; +0.108] * |
| 1 | 0.761 | 0.732 | 0.818 | 0.802 | 0.826 | +0.065 [+0.018; +0.112] * | +0.095 [+0.046; +0.145] * | +0.024 [-0.004; +0.055] |
| 2 | 0.827 | 0.831 | 0.825 | 0.829 | 0.868 | +0.041 [+0.003; +0.082] * | +0.037 [+0.001; +0.076] * | +0.040 [+0.008; +0.072] * |
| 3 | 0.705 | 0.707 | 0.723 | 0.742 | 0.820 | +0.115 [+0.066; +0.165] * | +0.113 [+0.066; +0.162] * | +0.079 [+0.036; +0.123] * |
| 4 | 0.794 | 0.797 | 0.795 | 0.798 | 0.854 | +0.059 [+0.014; +0.108] * | +0.057 [+0.014; +0.105] * | +0.056 [+0.015; +0.102] * |
| **TB** | 0.766 | 0.760 | 0.789 | 0.794 | 0.846 | 0.081 | 0.086 | 0.053 |

AUPRC. AUROC:

| Fold | A | B | D2 | J | **JF** |
|---|---|---|---|---|---|
| 0 | 0.899 | 0.895 | 0.932 | 0.931 | 0.941 |
| 1 | 0.903 | 0.897 | 0.928 | 0.929 | 0.933 |
| 2 | 0.914 | 0.924 | 0.921 | 0.915 | 0.936 |
| 3 | 0.871 | 0.871 | 0.875 | 0.891 | 0.913 |
| 4 | 0.921 | 0.913 | 0.916 | 0.916 | 0.930 |
| **TB** | 0.902 | 0.900 | 0.915 | 0.916 | 0.930 |

**Bỏ ảnh có dụng cụ kim loại** (98% ảnh có dụng cụ được gán nhãn gãy: lối tắt). AUPRC, bootstrap theo ảnh:

| Fold | Ảnh có dụng cụ (test) | A | B | D2 | J | **JF** | JF − B |
|---|---|---|---|---|---|---|---|
| 0 | 14 | 0.682 | 0.676 | 0.733 | 0.755 | 0.834 | +0.158 [+0.103; +0.213] * |
| 1 | 11 | 0.722 | 0.686 | 0.788 | 0.771 | 0.800 | +0.115 [+0.059; +0.175] * |
| 2 | 23 | 0.786 | 0.788 | 0.771 | 0.779 | 0.835 | +0.047 [-0.000; +0.097] |
| 3 | 27 | 0.615 | 0.623 | 0.637 | 0.664 | 0.775 | +0.152 [+0.091; +0.220] * |
| 4 | 24 | 0.725 | 0.731 | 0.726 | 0.729 | 0.810 | +0.079 [+0.019; +0.142] * |
| **TB** | | 0.706 | 0.701 | 0.731 | 0.739 | 0.811 | |

## 2. BTXRD (u xương): test / external

**test (cùng nguồn)**

| Fold | Có u: AUROC D2 / J / **JF** | Ác tính: AUROC D2 / J / **JF** | Chẩn đoán: acc D2 / J / **JF** | JF − J (u / ác tính / chẩn đoán) |
|---|---|---|---|---|
| 0 | 0.690 / 0.934 / 0.907 | 0.400 / 0.833 / 0.881 | 0.000 / 0.546 / 0.546 | -0.027 [-0.050; -0.006] * / +0.048 [-0.001; +0.100] / +0.000 [-0.035; +0.034] |
| 1 | 0.701 / 0.903 / 0.905 | 0.713 / 0.821 / 0.857 | 0.000 / 0.528 / 0.498 | +0.002 [-0.026; +0.029] / +0.036 [-0.002; +0.077] / -0.030 [-0.069; +0.008] |
| 2 | 0.639 / 0.902 / 0.893 | 0.454 / 0.828 / 0.876 | 0.000 / 0.517 / 0.509 | -0.009 [-0.028; +0.010] / +0.047 [+0.004; +0.091] * / -0.007 [-0.043; +0.028] |
| 3 | 0.593 / 0.896 / 0.912 | 0.663 / 0.824 / 0.889 | 0.000 / 0.535 / 0.520 | +0.017 [-0.003; +0.037] / +0.065 [-0.002; +0.145] / -0.015 [-0.062; +0.030] |
| 4 | 0.573 / 0.909 / 0.903 | 0.674 / 0.827 / 0.863 | 0.000 / 0.539 / 0.517 | -0.006 [-0.025; +0.014] / +0.036 [-0.040; +0.124] / -0.022 [-0.059; +0.014] |
| **TB** | 0.639 / 0.909 / 0.904 | 0.581 / 0.827 / 0.873 | 0.000 / 0.533 / 0.518 | |

**external (trung tâm khác)**

| Fold | Có u: AUROC D2 / J / **JF** | Ác tính: AUROC D2 / J / **JF** | Chẩn đoán: acc D2 / J / **JF** | JF − J (u / ác tính / chẩn đoán) |
|---|---|---|---|---|
| 0 | 0.723 / 0.842 / 0.813 | 0.400 / 0.610 / 0.614 | 0.000 / 0.284 / 0.287 | -0.029 [-0.054; -0.007] * / +0.005 [-0.034; +0.040] / +0.004 [-0.022; +0.032] |
| 1 | 0.799 / 0.785 / 0.803 | 0.536 / 0.568 / 0.620 | 0.000 / 0.270 / 0.274 | +0.018 [-0.010; +0.046] / +0.053 [+0.017; +0.088] * / +0.004 [-0.022; +0.029] |
| 2 | 0.728 / 0.802 / 0.806 | 0.490 / 0.588 / 0.594 | 0.000 / 0.264 / 0.289 | +0.004 [-0.016; +0.025] / +0.007 [-0.028; +0.042] / +0.025 [+0.002; +0.049] * |
| 3 | 0.648 / 0.761 / 0.802 | 0.522 / 0.567 / 0.627 | 0.000 / 0.264 / 0.274 | +0.041 [+0.010; +0.076] * / +0.060 [+0.014; +0.104] * / +0.010 [-0.019; +0.039] |
| 4 | 0.629 / 0.817 / 0.805 | 0.527 / 0.624 / 0.670 | 0.000 / 0.287 / 0.276 | -0.012 [-0.040; +0.015] / +0.046 [+0.001; +0.094] * / -0.011 [-0.034; +0.010] |
| **TB** | 0.705 / 0.801 / 0.806 | 0.495 / 0.591 / 0.625 | 0.000 / 0.274 / 0.280 | |

B không có cột BTXRD: B không học BTXRD và không có hộp u xương. D2 không học câu hỏi–đáp BTXRD (chỉ có hộp u từ YOLOv8), nên cột D2 cho thấy phần đóng góp của dữ liệu BTXRD; chẩn đoán của D2 = 0.

## 3. Lớp tin cậy (xác minh / từ chối trả lời)

Độ tin cậy đã hiệu chỉnh + điểm hộp của YOLOv8, khớp trên val, điểm vận hành ~80% câu được trả lời. Ô = độ phủ: độ chính xác tất cả → độ chính xác phần được trả lời.

| Fold | Nhánh | Gãy (test) | U (test) | U (external) | Ác tính (test) | Ác tính (external) | Chẩn đoán (test) | Chẩn đoán (external) |
|---|---|---|---|---|---|---|---|---|
| 0 | J | 79%: 0.903 → 0.973 | 83%: 0.864 → 0.930 | 81%: 0.792 → 0.837 | 76%: 0.848 → 0.917 | 62%: 0.743 → 0.850 | 70%: 0.550 → 0.630 | 83%: 0.287 → 0.301 |
| 0 | JF | 82%: 0.939 → 0.968 | 83%: 0.860 → 0.902 | 74%: 0.783 → 0.835 | 74%: 0.885 → 0.940 | 64%: 0.720 → 0.848 | 74%: 0.543 → 0.636 | 77%: 0.297 → 0.312 |
| 1 | J | 80%: 0.905 → 0.962 | 84%: 0.838 → 0.892 | 81%: 0.739 → 0.780 | 75%: 0.851 → 0.930 | 72%: 0.807 → 0.843 | 77%: 0.532 → 0.582 | 83%: 0.284 → 0.283 |
| 1 | JF | 78%: 0.917 → 0.965 | 78%: 0.855 → 0.926 | 72%: 0.782 → 0.813 | 80%: 0.866 → 0.930 | 78%: 0.805 → 0.850 | 72%: 0.506 → 0.574 | 71%: 0.278 → 0.299 |
| 2 | J | 80%: 0.913 → 0.958 | 81%: 0.842 → 0.895 | 77%: 0.764 → 0.814 | 78%: 0.866 → 0.914 | 71%: 0.768 → 0.830 | 73%: 0.532 → 0.604 | 78%: 0.289 → 0.313 |
| 2 | JF | 82%: 0.939 → 0.970 | 85%: 0.843 → 0.886 | 74%: 0.757 → 0.807 | 79%: 0.840 → 0.925 | 77%: 0.797 → 0.834 | 74%: 0.528 → 0.598 | 84%: 0.295 → 0.307 |
| 3 | J | 81%: 0.901 → 0.954 | 85%: 0.821 → 0.906 | 81%: 0.717 → 0.806 | 75%: 0.851 → 0.921 | 74%: 0.741 → 0.819 | 77%: 0.532 → 0.599 | 83%: 0.272 → 0.296 |
| 3 | JF | 79%: 0.913 → 0.956 | 83%: 0.862 → 0.916 | 82%: 0.781 → 0.812 | 75%: 0.859 → 0.941 | 71%: 0.808 → 0.837 | 77%: 0.520 → 0.617 | 83%: 0.280 → 0.301 |
| 4 | J | 83%: 0.901 → 0.966 | 80%: 0.843 → 0.919 | 79%: 0.767 → 0.829 | 74%: 0.851 → 0.935 | 72%: 0.795 → 0.842 | 76%: 0.550 → 0.618 | 82%: 0.289 → 0.309 |
| 4 | JF | 83%: 0.936 → 0.965 | 86%: 0.836 → 0.889 | 80%: 0.777 → 0.801 | 75%: 0.862 → 0.951 | 72%: 0.799 → 0.850 | 71%: 0.509 → 0.618 | 76%: 0.284 → 0.301 |

Từ chối trả lời đổi độ phủ lấy độ chính xác; nó không làm tăng AUROC.

## 4. Kết quả bổ trợ trên fold 0

**a. J với 3 seed (42 / 43 / 44)**

| Seed | FracAtlas AUPRC | U AUROC test / ext | Ác tính AUROC test / ext | Chẩn đoán acc test / ext |
|---|---|---|---|---|
| 42 | 0.799 | 0.934 / 0.842 | 0.833 / 0.610 | 0.546 / 0.284 |
| 43 | 0.794 | 0.927 / 0.830 | 0.785 / 0.576 | 0.535 / 0.282 |
| 44 | 0.793 | 0.934 / 0.831 | 0.839 / 0.573 | 0.513 / 0.280 |

Seed chỉ đổi thứ tự lô BTXRD và dropout, nên độ dao động theo seed là cận dưới.

**b. Đối chứng KG**

| Nhánh | FracAtlas AUPRC | U AUROC test / ext | Ác tính AUROC test / ext | Chẩn đoán acc test / ext |
|---|---|---|---|---|
| J | 0.799 | 0.934 / 0.842 | 0.833 / 0.610 | 0.546 / 0.284 |
| K (KG prompt) | 0.814 | 0.934 / 0.840 | 0.825 / 0.594 | 0.535 / 0.268 |
| K0 (KG rỗng) | 0.800 | 0.937 / 0.838 | 0.829 / 0.605 | 0.539 / 0.282 |

K − J (test): u -0.000 [-0.007; +0.006], ác tính -0.008 [-0.029; +0.013], chẩn đoán -0.011 [-0.031; +0.007]

K − J (external): u -0.002 [-0.011; +0.008], ác tính -0.016 [-0.031; -0.002] *, chẩn đoán -0.015 [-0.036; +0.006]

**c. Thông tin tham chiếu: vị trí tổn thương (chẩn đoán BTXRD, J, chấm theo ứng viên)**

| Cách dùng | Acc test (Δ so với chỉ ảnh) | Acc external (Δ) |
|---|---|---|
| image only (plain question) | 0.550 | 0.287 |
| image + data prior (site) | 0.632 (+0.082 [+0.016; +0.153] *) | 0.375 (+0.088 [+0.051; +0.126] *) |
| metadata only (site) | 0.584 (+0.033 [-0.071; +0.141]) | 0.345 (+0.057 [-0.006; +0.119]) |
| image + data prior (age + sex + site + region + view) | 0.621 (+0.071 [+0.004; +0.140] *) | 0.387 (+0.100 [+0.061; +0.141] *) |
| image + KG prior (age, site edges) | 0.528 (-0.022 [-0.056; +0.008]) | 0.287 (+0.000 [-0.023; +0.023]) |
| wording: age, sex, site in the question | 0.494 (-0.056 [-0.104; -0.011] *) | 0.268 (-0.019 [-0.046; +0.008]) |
| majority class (no image, no metadata) | 0.446 (-0.104 [-0.164; -0.046] *) | 0.239 (-0.048 [-0.101; +0.004]) |

Vị trí tổn thương không được dùng cho câu hỏi có u hay không: BTXRD chỉ ghi vị trí cho ảnh có u.

**d. Grounding (D2, fold 0, test): V4 có dựa vào vị trí hộp không?**

| Mức (ảnh bình thường có hộp, model_val) | AUPRC model_val / test | Precision hộp @0,5 | Ảnh gãy có hộp trúng | Ảnh gãy không hộp (độ nhạy) | Placebo ΔAUPRC | Oracle ΔAUPRC |
|---|---|---|---|---|---|---|
| 0.02 | 0.871 / 0.852 | 0.655 | 0.657 | 0.322 (0.348) | -0.005 [-0.018; +0.009] | +0.001 [-0.003; +0.005] |
| 0.15 | 0.858 / 0.811 | 0.348 | 0.741 | 0.182 (0.308) | -0.085 [-0.124; -0.049] * | -0.001 [-0.012; +0.011] |
| 0.284 (tỉ lệ của Faster R-CNN) | 0.857 / 0.784 | 0.201 | 0.804 | 0.077 (0.273) | -0.051 [-0.086; -0.018] * | +0.006 [-0.007; +0.019] |
| max | 0.854 / 0.775 | 0.155 | 0.825 | 0.028 (0.250) | -0.071 [-0.107; -0.034] * | +0.010 [-0.002; +0.024] |

Điểm F1@0,5 tốt nhất của riêng YOLOv8 trên model_val: ngưỡng 0.111, 3.8% ảnh bình thường có hộp.

D2 mức 0,02 so với D2 (ngưỡng theo Faster R-CNN), test AUPRC: fold 0 +0.068 [+0.021; +0.120] *, fold 1 +0.011 [-0.013; +0.035].

**e. Chọn ngưỡng BTXRD (chỉ suy luận, JF fold 0, BTXRD model_val; không nhìn test / external)**

| Tỉ lệ ảnh bình thường có hộp | U AUROC | Ác tính AUROC | Chẩn đoán acc |
|---|---|---|---|
| 0.02 | 0.838 | 0.837 | 0.530 |
| 0.05 | 0.836 | 0.829 | 0.530 |
| 0.10 **(chọn)** | 0.847 | 0.850 | 0.530 |
| 0.15 | 0.847 | 0.839 | 0.515 |
| 0.30 | 0.821 | 0.835 | 0.515 |

Quy tắc khai báo trước: max tumour AUROC on BTXRD model_val, fold 0, JF trained with BTXRD level 0.02; inference only; ties -> stricter level.

## 5. Hạn chế cần ghi trong luận văn

- **Proposal BTXRD lúc huấn luyện V4 là in-sample.** Ba detector cross-fit của mỗi fold chỉ chia FracAtlas train; BTXRD train được đưa nguyên vào cả ba. Vì vậy ảnh BTXRD lúc huấn luyện nhận hộp từ detector đã học chính ảnh đó (tỉ lệ ảnh có u được đặt hộp cao hơn lúc test). V4 có thể tin hộp u quá mức, nhất là trên tập external. Sửa: chia luôn BTXRD train trong 3 detector cross-fit (bản IEEE).
- Mỗi fold chỉ một seed. Seed chỉ đổi thứ tự lô BTXRD và dropout, nên độ dao động là cận dưới.
- Mức ngưỡng FracAtlas 0,02 được chọn trên model_val của fold 0 rồi cố định cho các fold khác; ở fold 1, model_val lại chuộng mức khác. Ngưỡng BTXRD chọn trên model_val fold 0 chỉ ở bước suy luận; JF được huấn luyện với BTXRD mức 0,02.
- Detector YOLOv8s, ảnh 640, 60 epoch; chưa tinh chỉnh kích thước mô hình hay độ phân giải ảnh.
- Lợi ích của YOLOv8 đến từ việc có hộp với độ tin cậy cao (placebo ở mức 0,02 không làm giảm), không phải từ việc V4 định vị đúng ổ gãy.
- Ảnh có dụng cụ kim loại là lối tắt cho nhãn gãy; kết quả khi bỏ nhóm này nằm ở bảng 1.
- Chẩn đoán tên bệnh BTXRD còn yếu, nhất là trên tập external; bốn lớp hiếm không bao giờ được dự đoán.
- Fold 4 chưa chạy (tạm dừng theo phạm vi đóng băng 20/11).
