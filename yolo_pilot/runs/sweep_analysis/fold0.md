# Quét ngưỡng YOLOv8, fold 0 (tự sinh bởi code/sweep_analysis.py)

## 1. Detector riêng (model_val, chọn theo F1 của chính YOLOv8, không dùng Faster R-CNN)

- Max f1@0.5: ngưỡng 0.1114, tỉ lệ ảnh bình thường có hộp 0.038 (model_val); trên test: precision@0.5 0.590, recall@0.5 0.520, ảnh gãy có hộp trúng (IoU≥0.3) 0.685
- Max f1@0.3: ngưỡng 0.0468, tỉ lệ ảnh bình thường có hộp 0.053 (model_val); trên test: precision@0.5 0.488, recall@0.5 0.537, ảnh gãy có hộp trúng (IoU≥0.3) 0.706

## 2. Theo mức quét (test)

| Mức | Ngưỡng outer | Ảnh BT có hộp | Hộp sai / ảnh BT | Precision hộp @0.3 / @0.5 | Recall hộp @0.3 / @0.5 | Ảnh gãy có hộp trúng @0.3 | V4 AUPRC model_val | V4 AUPRC test | AUROC test |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.2263 | 0.015 | 0.02 | 0.852 / 0.655 | 0.610 / 0.492 | 0.657 | 0.871 | 0.852 | 0.936 |
| 0.15 | 0.0135 | 0.138 | 0.17 | 0.457 / 0.348 | 0.684 / 0.554 | 0.741 | 0.858 | 0.811 | 0.939 |
| 0.284 | 0.0024 | 0.309 | 0.53 | 0.278 / 0.201 | 0.751 / 0.599 | 0.804 | 0.857 | 0.784 | 0.932 |
| max | 0.0010 | 0.424 | 0.84 | 0.215 / 0.155 | 0.774 / 0.616 | 0.825 | 0.854 | 0.775 | 0.926 |

Mức chọn trên model_val: **0.02**

## 3. Grounding của câu trả lời (test, ảnh gãy)

| Mức | Hộp trúng / trượt / không hộp | Độ nhạy: trúng / trượt / không | TP có hộp trúng | Đúng có căn cứ / ảnh gãy | Placebo ΔAUPRC [CI] | Oracle vị trí ΔAUPRC [CI] |
|---|---|---|---|---|---|---|
| 0.02 | 0.66 / 0.02 / 0.32 | 0.97 / 1.00 / 0.35 | 0.83 | 0.64 | -0.005 [-0.018; +0.009] | +0.001 [-0.003; +0.005] |
| 0.15 | 0.74 / 0.08 / 0.18 | 0.82 / 0.73 / 0.31 | 0.84 | 0.61 | -0.085 [-0.124; -0.049] | -0.001 [-0.012; +0.011] |
| 0.284 | 0.80 / 0.12 / 0.08 | 0.75 / 0.47 / 0.27 | 0.89 | 0.60 | -0.051 [-0.086; -0.018] | +0.006 [-0.007; +0.019] |
| max | 0.83 / 0.15 / 0.03 | 0.74 / 0.48 / 0.25 | 0.89 | 0.61 | -0.071 [-0.107; -0.034] | +0.010 [-0.002; +0.024] |
