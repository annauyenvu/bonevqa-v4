# Giao thức thí nghiệm bổ sung cho bài IEEE Access

Khóa ngày 2026-11-02, trước khi đọc bất kỳ kết quả test nào của các điều kiện dưới đây. Mọi nhánh mới là đối chứng và tái lập: không có nhánh nào được chọn làm mô hình cuối, không chỉnh ngưỡng, không thêm epoch. Tập test đã được đọc nhiều lần trong các nghiên cứu trước, nên đây không phải test mù; bài báo phải nêu rõ điều này.

## Phần A. Huấn luyện lại (suite `access_v1`, `experiments/v5/study_v5.py`)

Công thức giữ nguyên nghiên cứu niêm phong: khởi tạo PRE theo fold, 2 epoch, LAST, AdamW 1e-4, batch 4 × tích lũy 4, detector và proposal niêm phong. Seed 43, 5 fold, huấn luyện trên MPS, đánh giá trên CPU, mỗi job có gate gradient và replay tiến trình mới.

| Nhánh | Tín hiệu có vùng theo detector | Nội dung vùng | Ý nghĩa |
|---|---|---|---|
| B | Không | Không | Baseline |
| V4_flag | Có | Không, vector cố định theo slot | Chỉ tính sẵn có |
| V4_grid | Không, ba ô cố định trên mọi ảnh | Có | Chỉ độ phân giải cục bộ |
| V4_placebo | Có | Có, ở vị trí đã dịch | Sai vị trí |
| V4 | Có | Có | Thiết kế đầy đủ |

B fold 0 seed 43 dùng lại lần chạy đã có trong `v5_studies/factorial_fold0` vì cùng công thức, cùng thiết bị, cùng seed.

**Endpoint chính:** AP gãy xương trên câu canonical, gộp 5 fold, 4.013 ảnh. Khoảng tin cậy: bootstrap ghép cặp 2.000 lần theo component, seed 20261103. Hiệu chỉnh Holm trên năm so sánh chính ở mức 0,05.

| Mã | So sánh | Câu hỏi |
|---|---|---|
| C1 | V4 − B | Lợi ích có tái lập ở seed độc lập không |
| C2 | V4_flag − B | Riêng tín hiệu sẵn có có tạo lợi ích không |
| C3 | V4_grid − B | Crop không kèm tín hiệu detector có tạo lợi ích không |
| C4 | V4 − V4_flag | Nội dung crop có thêm gì sau khi đã có tín hiệu sẵn có không |
| C5 | V4 − V4_placebo | Vị trí đúng có thêm gì không |

**Diễn giải khóa trước:**
- C2 dương, C4 không dương: lợi ích chủ yếu là tín hiệu sẵn có.
- C4 dương: nội dung crop có đóng góp khi huấn luyện lại, trái với phép can thiệp sau huấn luyện, và phải báo cáo như vậy.
- C3 dương: crop có giá trị độc lập với detector.
- C1 không dương: lợi ích V4 − B không tái lập ở seed 43, và luận điểm chính phải hạ xuống.
- CI chứa 0 không được diễn giải là tương đương.

**Endpoint phụ, không hiệu chỉnh, chỉ mô tả:** AUROC, sensitivity, specificity, BA, Brier ở ngưỡng 0,5; QA exact chung, câu đóng, câu mở; hiệu theo fold.

**Giới hạn đã biết:** chỉ một seed cho các đối chứng; detector giữ nguyên nên chưa có biến thiên detector; placebo giữ tính sẵn có nhưng không giữ phân phối vị trí giải phẫu; lưới ba ô đổi số crop của mọi ảnh nên tăng chi phí tính toán.

## Phần B. Không huấn luyện lại (checkpoint niêm phong seed 42, B và V4)

1. **Thống kê đầy đủ** (`access_stats_v1.py`): AP, AUROC, Brier, ECE 10 bin, độ dốc hiệu chuẩn, McNemar cho đáp án câu canonical và QA toàn bộ, hiệu theo fold.
2. **Nhóm con:** vùng cơ thể, có dụng cụ kim loại, góc chụp, số vị trí gãy. Chỉ mô tả, không dùng để chọn gì; tỷ lệ có proposal báo cùng từng nhóm.
3. **Câu trả lời mở:** exact, F1 token, BLEU-1 theo họ câu hỏi.
4. **Detector:** recall ở IoU 0,1, 0,3, 0,5 theo fold và vùng cơ thể; số hộp sai trên ảnh không gãy.
5. **Diễn đạt lại câu hỏi** (`access_paraphrase_v1.py`): sáu câu dưới đây, cố định trước khi chạy, không xuất hiện trong TRAIN nếu kiểm tra xác nhận; AP, sensitivity, specificity, tỷ lệ đồng ý với câu canonical.
   - `Does this X-ray show a fracture?`
   - `Is a fracture present in this image?`
   - `Is any bone broken in this radiograph?`
   - `Can a fracture be seen on this film?`
   - `Is this X-ray positive for fracture?`
   - `Is there no fracture in this X-ray?` (phủ định, đáp án đúng đảo chiều)
6. **Baseline chỉ dùng ảnh** (`access_image_baselines_v1.py`): BiomedCLIP đóng băng + hồi quy logistic (C = 1); MobileNetV3-Large khởi tạo ImageNet, 10 epoch cố định, lấy epoch cuối. Chỉ học trên TRAIN của từng fold, không chọn gì trên test.

## Phần C. Chưa làm được trong giao thức này

- **Kiểm định ngoài ở miền người lớn nhiều vùng cơ thể:** Phần D chỉ phủ cổ tay trẻ em.
- **Thẩm định bởi bác sĩ:** cộng tác viên chuyên môn đọc lại câu trả lời và ca lỗi.
- **Nhóm theo bệnh nhân:** FracAtlas không có mã bệnh nhân; chỉ có component theo kiểm tra trùng lặp.
- **So sánh 5 fold với phương pháp đã công bố:** chuyển sang Phần E theo yêu cầu ngày 2026-11-03.

## Phần D. Kiểm định ngoài trên GRAZPEDWRI-DX

Khóa ngày 2026-11-02, trước khi chạy mô hình và trước khi xem phân phối nhãn của dữ liệu này. Nguồn: Figshare, DOI 10.6084/m9.figshare.14825193.v2, giấy phép CC BY 4.0. Đây là X-quang cổ tay trẻ em, khác FracAtlas về độ tuổi, vùng cơ thể và nơi chụp, nên đo khả năng chuyển miền, không phải hiệu năng dự kiến khi triển khai cho người lớn.

- **Mô hình:** năm cặp checkpoint niêm phong B và V4 seed 42 cùng detector theo fold; không huấn luyện lại, không tinh chỉnh, không hiệu chuẩn lại, ngưỡng 0,5, câu canonical.
- **Mẫu:** 2.000 ảnh ngẫu nhiên, mỗi bệnh nhân một ảnh, seed 20261103, chọn trước khi đọc nhãn.
- **Endpoint chính:** AUROC và AP của V4 và B, tính cho từng checkpoint rồi lấy trung bình; ΔAP và ΔAUROC của V4 − B với bootstrap theo ảnh 2.000 lần, seed 20261103. Vì tỷ lệ gãy khác FracAtlas, AP phải báo kèm tỷ lệ gãy của mẫu.
- **Endpoint phụ, chỉ mô tả:** sensitivity, specificity, Brier, ECE; tỷ lệ có proposal theo nhãn; kết quả sau khi bỏ ảnh có kim loại; tiền xử lý giữ đúng `canonicalize_raster`.
- **Diễn giải khóa trước:** ΔAP có CI dương nghĩa là lợi ích của V4 chuyển được sang miền này. CI chứa 0 không được diễn giải là tương đương. ΔAP âm hoặc tỷ lệ proposal giống nhau giữa hai nhãn nghĩa là tín hiệu detector không chuyển miền. Kết quả nào cũng được báo cáo, không đổi cấu hình sau khi xem.
- **Bổ sung kỹ thuật, ghi trước khi chạy mô hình và chưa xem phân phối nhãn:** toàn bộ 150 ảnh kiểm tra ngẫu nhiên là PNG xám 16 bit, dùng hết dải (max 65535, p99 trung vị 46640). `convert("RGB")` của PIL sẽ cắt bão hòa kiểu này, nên ảnh được nhân tuyến tính 255/65535 về 8 bit rồi mới qua `canonicalize_raster`. Không cân bằng histogram, không cửa sổ hóa. Bốn ảnh xem bằng mắt có xương sáng trên nền tối. FracAtlas là ảnh 8 bit, nên đây là một sai khác kỹ thuật giữa hai miền và bài báo phải nêu.

## Phần E. So sánh 5 fold với các kiểu prompt đã công bố (suite `access_v1_methods`)

Khóa ngày 2026-11-03, trước khi chạy fold 1 đến 4 của các nhánh này. Kết quả fold 0 đã được đọc từ trước, nên phần này không mù với fold 0. Không chọn nhánh nào làm mô hình cuối, không đổi cấu hình so với fold 0.

- **Nhánh:** V (kiểu FAVP), P (kiểu LaPA), L (kiểu Localization Lens), đúng cấu hình `FACTORS` trong `experiments/v5/study_v5.py`. Đây là các kiểu prompt được chuyển vào khung BoneVQA, không phải mã gốc của tác giả, và bài báo phải gọi chúng là bản thích nghi. Sai khác đã biết: V dùng proposal AVPC từ SAM-Med2D; P dùng đồ thị xương dựng từ TRAIN thay cho SLAKE; L chỉ dùng mặt nạ MedSAM vì SAM-H dừng ở 753/4.013 ảnh.
- **Huấn luyện:** seed 42, fold 1 đến 4, huấn luyện trên MPS, đánh giá trên CPU, gate và replay như Phần A. Fold 0 dùng lại `factorial_fold0` cùng công thức và seed. Chạy sau khi Phần A xong, một worker.
- **Đối chứng chính:** B và V4 niêm phong seed 42, 5 fold. **Đối chứng độ nhạy:** B và V4 seed 43 của Phần A, cùng thiết bị MPS nhưng khác seed.
- **Endpoint:** AP canonical gộp 5 fold; hiệu với B và với V4 cùng CI bootstrap ghép cặp 2.000 lần theo component, seed 20261103. Đây là so sánh phụ, chỉ mô tả, không thuộc họ Holm của Phần A. AUROC, sensitivity, specificity, BA, Brier, QA báo kèm.
- **Diễn giải khóa trước:** chỉ kết luận trong bản thích nghi và trên FracAtlas, không kết luận phương pháp gốc kém hơn. CI chứa 0 không được diễn giải là tương đương. Một seed nên chưa có biến thiên seed cho các nhánh này.

## Nhật ký sai lệch

- **2026-11-03:** khi thử mã phân tích Phần E trên fold 0, đầu ra in cả AP canonical seed 43 của Phần A trên fold 0 (B 0,691, V4 0,734), trước khi đủ 5 fold. Đây là lần xem sớm một phần của C1, không có so sánh nào khác bị xem. Không có cấu hình, ngưỡng hay kế hoạch nào đổi sau đó. Bài báo phải nêu điều này nếu thuật lại quy trình.
- **2026-11-05 11:1x:** theo yêu cầu xem kết quả sơ bộ, chạy `methods` trên fold 0 và 1 (`methods_interim_folds01.json`) khi fold 2 đến 4 của Phần E chưa xong. Không có cấu hình, ngưỡng hay kế hoạch nào đổi sau đó; kết quả chính vẫn là phân tích đủ 5 fold.
- **2026-11-06 10:3x:** sau khi xem kết quả Phần F, phát hiện lỗi mã trong `analyze`: với câu phủ định đi qua router, `p_yes` đã là p_fracture của fracture head nhưng bị lật thêm thành 1 − p, nên AP ra khoảng 0,10. Đã sửa để chỉ lật ở nhánh `current` (head đóng), đúng với quy tắc F1 đã khóa; không chạy lại mô hình, không đổi câu hỏi, ngưỡng hay router. Đầu ra lỗi được giữ ở `routing/results_buggy_negation.json`.
- **2026-11-05 23:3x:** lần xem sớm thứ hai, `methods` trên fold 0 đến 2 (`methods_interim_folds012.json`), khi fold 3 và 4 chưa xong. Không có gì đổi sau đó.

## Phần F. Định tuyến câu hỏi theo ý nghĩa và ngưỡng chọn trên calibration (`experiments/access_routing_v1.py`)

Khóa ngày 2026-11-03, trước khi chạy bất kỳ phần nào dưới đây. Theo yêu cầu, phần này chỉ chạy sau khi Phần A và Phần E xong. Không huấn luyện lại, dùng checkpoint niêm phong seed 42 của B và V4, chạy trên CPU.

**Lý do:** `src/endpoints.py` chỉ đưa đúng mẫu câu canonical vào fracture head. Mọi câu diễn đạt lại, kể cả câu phủ định, đi qua head đóng, nên Phần B.5 đo head đóng chứ không đo endpoint gãy xương. Ngưỡng 0,5 lại được khóa mà không chọn trên tập nào.

**F1. Router theo ý nghĩa**, quy tắc viết trước khi chạy:
- Câu hỏi được đưa vào fracture head khi thỏa cả ba điều kiện:
  - là câu đóng theo `is_polar_question`;
  - có một trong các từ `fracture(s)`, `fractured`, `broken`, `break`;
  - không có từ chỉ vùng cơ thể, số lượng hay vị trí, theo danh sách cố định trong mã.
- Câu có `no`, `not`, `n't`, `without`, `absent` hoặc `free of` được coi là phủ định: điểm gãy xương là p_fracture, còn đáp án "yes" ứng với 1 − p_fracture.
- **Kiểm tra trước khi chạy:**
  - Đã đạt: trong 28 mẫu câu của dữ liệu, router chỉ nhận câu canonical, không nhận mẫu nào khác.
  - Chạy đầu tiên trong hàng đợi: trên 80 ảnh fold 0, câu canonical phải cho cùng p_yes với dự đoán niêm phong, sai lệch dưới 1e-4. Nếu không đạt thì dừng toàn bộ Phần F.
- **Câu hỏi:**
  - Sáu câu cũ của Phần B.5. Router được viết sau khi đã thấy kết quả của các câu này, nên chúng chỉ để mô tả.
  - Sáu câu mới khóa tại đây, chưa từng chạy và không có trong TRAIN. Kết quả trên sáu câu mới là kết quả chính của F1:
    - `Do you see a fracture on this radiograph?`
    - `Is there a broken bone in this image?`
    - `Does this radiograph demonstrate a fracture?`
    - `Is a bone fracture visible here?`
    - `Is this X-ray free of fractures?` (phủ định)
    - `Is there not any fracture on this film?` (phủ định)
- **Endpoint:** AP theo từng câu khi dùng router so với định tuyến hiện tại, cho cả B và V4; V4 − B khi dùng router. CI bootstrap theo component 2.000 lần, seed 20261103. Endpoint phụ: tỷ lệ đồng ý với câu canonical, sensitivity, specificity. Chỉ mô tả, không thuộc họ Holm.

**F2. Ngưỡng và hiệu chuẩn trên vai trò calibration:**
- **Ngưỡng:** ngưỡng Youden được chọn trên câu canonical ở vai trò `calibration` của từng fold, rồi áp nguyên lên test của fold đó.
- **Hiệu chuẩn:** Platt cũng được fit trên `calibration`, rồi áp lên test.
- **Kiểm định ngoài:** cùng ngưỡng và cùng Platt của FracAtlas được áp nguyên lên kiểm định ngoài, không chọn lại trên GRAZPEDWRI-DX.
- **Báo cáo:** sensitivity, specificity, BA ở ngưỡng 0,5 và ở ngưỡng Youden; ECE và Brier trước và sau Platt.
- **Không thay thế kết quả chính:** ngưỡng 0,5 vẫn là kết quả chính của luận văn và của bài báo, còn F2 chỉ là phân tích bổ sung.

**Diễn giải khóa trước:**
- **Router tốt hơn trên câu mới:** nghĩa là việc định tuyến theo mẫu chính xác là một nguyên nhân gây kém bền.
- **Router vẫn kém trên câu mới:** nghĩa là chính fracture head nhạy với cách hỏi.
- **Phủ định:** câu phủ định đúng nhờ quy tắc đảo của router, không phải mô hình hiểu phủ định, và bài báo phải nói rõ điều này.
- **Không kết luận tương đương:** CI chứa 0 không được diễn giải là tương đương.
