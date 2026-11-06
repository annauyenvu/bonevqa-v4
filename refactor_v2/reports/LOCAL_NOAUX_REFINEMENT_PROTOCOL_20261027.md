# B và visual local tokens không auxiliary loss

Giao thức khóa trước kết quả mới, ngày 27/10/2026. Người dùng cho phép thực hiện liên tục, sửa lỗi kỹ thuật và báo tiến độ mỗi giờ. Mục tiêu là kiểm chứng cải thiện, không lặp lựa chọn cấu hình đến khi có số dương.

## Phạm vi

Năm fold FracAtlas đã dùng cho baseline v3, cùng manifests TRAIN/model_val/calibration/test và vocabulary TRAIN. Các tập đã được xem trong development; không gọi là clinical confirmation độc lập. Không thử lại ROCO/PMC như test cho một model chuyên FracAtlas. Báo cáo cuối bao gồm QA đầy đủ model_val/test từng fold, canonical TRAIN/calibration, pooled outer 4.013 ảnh/17.998 QA và đối chiếu baseline lịch sử.

Mỗi fold có B mới và local_real_noaux mới, seed42, fresh cùng PREtask của fold, hai epoch theo schedule grouped baseline đã khóa. Có thêm local_placebo_noaux tại fold0, cùng schedule/compute của local_real. Tổng11 quality runs, không chọn best epoch. Đây là năm fold với một seed, không phải năm seed. B mới giúp kiểm tra runner và so sánh đồng điều kiện; baseline lịch sử giữ nguyên, được báo riêng.

Nhánh local sử dụng VNextModel đã được kiểm thử: tối đa3 automatic crops, mỗi crop224px, pooling2×2, question-conditioned attention và residual gate. `vnext_aux_weight=0.0`; các loss baseline khác giữ nguyên. Không thêm Lens/latent/textual prompt, không GT ROI ở public inference. Số bước không tương đương FLOPs. Local route thêm visual/fusion calls và dropout draws, phải công bố.

Detector fold0 LAST640 đã train trên fold0TRAIN được tái dùng sau kiểm tra hash/training IDs. Fold1–4 train detector riêng từ cached published initialization: FasterRCNN MobileNet320 architecture, effective640, hai epoch, SGD0.005/momentum0.9/wd0.0005/clip5/cosine2; positive-anchored batch4 theo seed42. Không dùng checkpoint detector fold0 cho outer fold khác. Proposal score0.05/top3/canonical coordinates, không thay threshold theo kết quả. Có thể reuse proposal cũ chỉ khi image hash và detector hash khớp; không annotation tại infer. Placebo dùng helper hình học đã có: chọn vị trí trên lưới hữu hạn để giảm overlap với proposal ban đầu rồi tăng khoảng dịch; giữ kích thước/số lượng/availability và không dùng GT để chọn vị trí. Proposal được làm tròn pixel như helper cũ; độ chênh hình học làm tròn phải được lưu.

## Train và kỹ thuật

AdamW lr1e-4, betas(0.9,0.98), eps1e-8, wd0.01, batch4, accum4, clip1; cosine và warmup5% theo src.train baseline. Giữ denominator4 của accumulation kể cả partial group để match historical recipe; số partial microbatch và effect được công bố. Hai epoch, LAST2; không đổi seed/loss/epoch/threshold sau xem assessment. Scheduled gate mỗi B/local/placebo: hai update từ TRAIN thật, ít nhất một batch local có proposal, gradient local answer-route, finite/frozen invariance, fresh replay. Fresh initialization quality không dùng checkpoint gate.

Sau train, mỗi arm trích feature fusion canonical TRAIN khi eval mode; fit bản sao fracture head với AdamW1e-4/betas0.9,0.98/wd0.01/batch64/clip1,10epoch cố định, không chọn best. Đây là secondary factorial head-refit on/off, không chỉnh head từ model_val/test. Backbone không đổi. Fit head cũng áp dụng B để tránh so sánh không cân bằng.

## Operating point và endpoints

Primary: pooled paired outer raw fracture AP local−B. Report BA/sensitivity/specificity ở0.5 và ở operating threshold chọn riêng trên calibration negatives để empirical specificity≥0.90, chọn threshold nhỏ nhất thỏa và xử lý ties bằng nextafter. Không fit threshold từ test. Head-refit là secondary, các target/recipe cố định trước kết quả; không tự chọn variant thắng trên outer.

Paired bootstrap2000 seed20261105 theo known component trong từng fold, giữ model/head/threshold cố định. Report overall/CLOSED/OPEN exact, AUROC/AP/Brier/ECE/confusion, image/proposal coverage, thời gian/RAM và raw outputs. Ưu thế đủ để đề nghị thay baseline chỉ khi primary AP delta có CI95% hoàn toàn dương và calibrated-operating BA delta có CI hoàn toàn dương, sensitivity delta dương, outer specificity không giảm quá2pp, overallQA không giảm quá0.5pp. Các điều kiện này là decision rule phát triển, không clinical approval. Placebo fold0 là semantic diagnostic; nếu local không vượt placebo phải công bố và không gọi local semantic benefit được xác nhận.

## Giới hạn thực thi

CPU8 offline runtime cũ; không install/download/cloud/upload. Reserve8GiB; child quality cap8h, detector2h, toàn workflow72h theo wall clock gồm sleep. Không train trùng. Lưu LAST quality, optimizer không dùng để tự resume; failure và partial giữ nguyên. Sửa lỗi cần version/review riêng trước retry, không tự xóa output. Không mở thêm seed/epoch hoặc thay scientific budget để tìm kết quả tốt. Hoàn tất đòi hỏi replays/audit/bootstrap/figures được xem thực tế/final interpretation/closure; nếu không tốt hơn, kết thúc với quyết định giữ B.
