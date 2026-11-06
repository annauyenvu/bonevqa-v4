# Lặp lại detector seed 43: tiêu chí ghi trước khi train

Ngày 2026-11-01. Hệ thống đã khóa, không chọn lại: B + B head-refit + cờ + điểm detector cao nhất, epoch 6, lớp nhất quán.

Khác duy nhất: seed sampler và dropout của phần tiếp tục (epoch 3-6) là 143+fold thay vì 42+fold. Điểm xuất phát là cùng detector epoch 2, nên đây là độ nhạy một phần với seed, không phải huấn luyện lại từ đầu.

Đọc test một lần sau khi có đủ 5 fold. Báo cáo nguyên trạng dù kết quả thế nào.

Tiêu chí lặp lại đạt khi cả ba điều sau đúng trên test gộp 5 fold:
1. AP seed 43 − V4 có cận dưới CI 95% > 0.
2. |AP seed 43 − AP seed 42| ≤ 0,015.
3. QA seed 43 − QA V4 có cận dưới CI 95% > 0.
