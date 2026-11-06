# Rà soát nhánh V, L, P, T từ v1 đến v5

Ngày 2026-11-01. Toàn bộ chạy trên CPU, chỉ đọc checkpoint, không huấn luyện lại và không sửa kết quả cũ.
Dữ liệu thô: `arms/`, `legacy/`, `availability.json`, `question_type.json`, `router.json`, `crop_attention.json`, `extra.json`, `static_checkpoints.json`.
Mã: `experiments/route_audit_*_v1.py`.

## 1. Kết luận

| Nhánh | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| **V** | Chỉ tới tiền tố LLM, không tới closed head | Gate 0,0028: gần như tắt | Gate 0,0041: gần như tắt | Có tác dụng, nhưng chủ yếu qua việc detector có đề xuất hay không | Tắt ở fusion (gate 0,002); HAG chỉ tác động nhẹ lên LLM |
| **L** | Có tác dụng, nhưng nội dung mask gần như không ảnh hưởng | Gate -0,0011: gần như tắt | Chỉ dùng khi huấn luyện | Không có | Chỉ dùng khi huấn luyện, ảnh hưởng gián tiếp qua LoRA |
| **P** | Có tác dụng và làm giảm kết quả | Không đo trên checkpoint đầy đủ | Gate 0,0005: gần như tắt | Không có | Có tác dụng nhưng không thêm AP; PF không có tác dụng |
| **T** | Luôn bật, chưa từng được tách riêng | Luôn bật | Có tác dụng ở câu trả lời, gần như không ở AP | Không có | Có tác dụng rất nhỏ, không thêm AP |

**Phát hiện quan trọng nhất cho bài báo:** mức tăng AP của V4 không phân biệt được về thống kê với việc ghép muộn B với tín hiệu "detector có đề xuất vùng và bao nhiêu vùng", không cần crop nào. Crop có đóng góp thêm nhưng phần này nhỏ hơn nhiều. Vì vậy số V4 > B vẫn đúng, nhưng không nên diễn giải là nhờ mô hình nhìn đúng vùng tổn thương.

Cách gọi v1/v2/v3 ở đây là quy ước của tôi vì dự án không định nghĩa chính thức v1, v2:
- **v1:** mã `exp4_multiseed/code`, các checkpoint `mandatory_experiments_20261011` và `exp4_multiseed`.
- **v2:** `fracatlas_clean_*_ce_v3`, kiến trúc có gate trong `prompt_mixer`.
- **v3:** các checkpoint P và T bounded, cùng bằng chứng V/L đã lưu.
- **v4:** `local_noaux_refinement_v1`.
- **v5:** `factorial_fold0`.

## 2. Vì sao V4 tăng AP

V4 chỉ áp `delta` khi ảnh có ít nhất một đề xuất, và tín hiệu này tự nó đã rất mạnh:

| Fold 0 | Có đề xuất, ảnh gãy | Có đề xuất, ảnh không gãy |
|---|---:|---:|
| TRAIN | 86,7% | 29,8% |
| Model-validation | 86,0% | 28,4% |
| Test | 84,6% | 32,3% |

Placebo giữ nguyên 100% việc ảnh nào có đề xuất, nên điều này giải thích vì sao placebo cũng tăng mạnh (AP 0,731 so với 0,740 của vùng thật).

**Ghép muộn, không crop, 5 fold gộp.** Hồi quy logistic trên logit của B, fit trên calibration và model-validation, đọc trên test:

| Mô hình | AP | Chênh lệch, CI 95% |
|---|---:|---|
| B | 0,716 | |
| V4 | 0,749 | so với B [+0,014; +0,051] |
| B + cờ có đề xuất | 0,741 | so với B [+0,010; +0,041]; V4 trừ mô hình này [-0,010; +0,025] |
| B + cờ + số vùng | 0,754 | so với B [+0,020; +0,056]; V4 trừ mô hình này [-0,022; +0,011] |

Khoảng tin cậy trong bảng này bootstrap theo ảnh, 2.000 lần. CI chứa 0 không chứng minh hai mô hình tương đương. Mô hình ghép muộn có dùng nhãn của calibration và model-validation, còn V4 không dùng.

**Can thiệp trên V4 đã huấn luyện, fold 0, test (803 ảnh):**

| Can thiệp | AP | Sens | Spec | ΔAP, CI 95% |
|---|---:|---:|---:|---|
| Nguyên bản | 0,741 | 65,0% | 93,8% | |
| Tắt delta | 0,629 | 28,7% | 98,8% | [-0,153; -0,071] |
| Crop thành nhiễu cùng phân phối | 0,722 | 62,2% | 93,8% | [-0,039; -0,000] |
| Dịch box sang vị trí khác | 0,706 | 58,0% | 93,6% | [-0,057; -0,013] |
| Lấy crop của ảnh khác | 0,571 | 43,4% | 92,9% | [-0,226; -0,117] |
| Bật cho mọi ảnh bằng 3 ô cố định | 0,662 | 74,1% | 82,3% | [-0,115; -0,043] |
| Box thật của tổn thương (ảnh có box) | 0,754 | 65,0% | 93,8% | [-0,007; +0,038] |

Cách đọc:
- Đường local thật sự được dùng khi suy luận: tắt delta làm AP giảm 0,112.
- Crop nhiễu vẫn giữ khoảng 83% phần đóng góp đó, box dịch chỗ giữ khoảng 69%. Nội dung và vị trí có đóng góp, nhưng là phần nhỏ.
- Bật cho mọi ảnh làm AP giảm 0,079, vì mất tín hiệu có/không có đề xuất. Điều này khớp với nhánh V: AVPC luôn có 3 vùng và gate của V học về 0,002, trong khi gate của V4 là 0,32–0,34.
- Lấy crop của ảnh khác tệ hơn nhiễu, vì nó đưa bằng chứng sai vào, kể cả sang những ảnh vốn không có đề xuất.

## 3. Tầng 1: code và đồ thị tính toán

**Gradient (v5, 8 hàng TRAIN thật, trọng số LAST, tắt dropout).**

| Nhánh | Module | Chuẩn gradient | Ghi chú |
|---|---|---:|---|
| V4 | `local_tokens.gate` | 0,826 | Sống |
| V4 | `local_tokens.attention` | 0,039 | Sống |
| V | `local_tokens.gate` | 0,0007 | Gần như chết |
| V | `local_tokens.attention` | 0,0018 | Gần như chết |
| VLPT | `local_tokens.gate` | 0,052 | Yếu |
| P | `v5_lf` | 0,670 | Sống |
| P | `v5_gm` | 0,223 | Sống |
| P | `v5_prior` (PF) | 0,009 | Yếu |
| L, loss QA | `v5_lens_fusion` | 0,545 | Chỉ từ loss Lens |
| L, loss vùng | `v5_lens_fusion` | 2,387 | Chỉ từ loss Lens |
| L, loss trả lời | `v5_lens_fusion` | Không có | Đúng thiết kế |

- `local_tokens.mil_score` và `aux_head` không có gradient. Đây là đúng thiết kế (auxiliary weight = 0), nhưng chúng là tham số thừa nằm trong checkpoint.
- T không có tham số riêng.
- Mọi nhánh đều có 2/26 tensor `visual.` không nhận gradient. Tôi chưa xác định từng tên; các tensor trainable không phải LoRA trong nhóm này là `logit_scale` và `head.proj.weight`.
- Chênh lệch tham số so với khởi tạo: LoRA thị giác của L và VLPT là 0,13, các nhánh khác khoảng 0,097. Lens thật sự đã thay đổi trọng số dùng chung.

**Hidden-state shift (độ lệch tương đối của `x_f` khi tắt nhánh, model-validation).**

| Nhánh | Can thiệp | Lệch `x_f` | Đổi đáp án fracture |
|---|---|---:|---:|
| V4 | Tắt delta | 13,7% | 10,9% |
| V | Tắt local và HAG | 0,04% | 0,0% |
| VLPT | Tắt V | 0,8% | 0,3% |
| T | Bỏ template | 13,6% | 0,6% |
| P | Tắt LF | 106,6% | 0,9% |
| P | Tắt PF | 0,5% | 0,0% |

**Attention mask: đạt.** Trên 9 nhánh model-validation và V4 test:
- Cùng một hàng chạy riêng hay trong lô: lệch logit tối đa dưới 1e-6.
- Đổi nội dung token đệm của text: lệch đúng 0.
- Đổi nội dung crop đệm: lệch đúng 0.
- Ảnh không có đề xuất: delta đúng 0.
- Template không bị cắt: tối đa 30 token, giới hạn 48 (text) và 56 (LLM).
- Mọi nhánh đều tái lập dự đoán đã lưu trước khi đo, lệch `p_yes` tối đa dưới 1e-5.

**Lỗi dây nối theo thế hệ:**
- **v1, V:** `f_i` không chứa token local, nên V không thể tới closed head. Tắt V cho lệch logit đúng 0 và đổi 0% closed. Mọi kết quả closed của "V" trong v1 đến từ khác biệt huấn luyện, không đến từ đường V khi suy luận.
- **v1, T:** T luôn bật, kể cả nhánh `none`. Do đó `none` của v1 không phải baseline không prompt.
- **v1, P và L:** P thay hẳn biểu diễn gộp, L thay hẳn luồng ảnh. Việc tắt chúng làm hỏng mô hình không chứng tỏ chúng hữu ích.
- **v2 và v3:** gate khởi tạo bằng 0 và gần như không mở được. Nhân gate 100 lần vẫn gần như không đổi kết quả (bảng ở mục 4).
- **v5, P:** Eq. 12 làm thành phần latent chiếm 98% độ lớn trước LayerNorm (chuẩn 161–166 so với 2,8 của mỗi thành phần θ, β).

## 4. Tầng 2: can thiệp khi suy luận

**v5, model-validation (321 ảnh, 57 ảnh gãy).** Kết quả v1–v3 ở bảng sau.

| Nhánh | Can thiệp | AP | ΔAP, CI 95% | Đổi closed | Đổi token đầu LLM |
|---|---|---:|---|---:|---:|
| V (0,753) | Tắt local | 0,753 | [0; 0] | 0% | 0% |
| V | Tắt HAG | 0,753 | [0; 0] | 0% | 3,3% |
| V | Tráo HAG giữa các mẫu | 0,753 | [0; 0] | 0% | 16,7% |
| V | Dịch box | 0,753 | [0; 0] | 0% | 1,7% |
| P (0,756) | Tắt LF | 0,745 | [-0,025; +0,002] | 8,0% | 8,3% |
| P | Nhiễu LF | 0,259 | [-0,591; -0,389] | 64,0% | 10,0% |
| P | Tráo LF giữa các mẫu | 0,202 | [-0,648; -0,419] | 60,0% | 11,7% |
| P | Tắt PF | 0,756 | [0; 0] | 0% | 0% |
| PT (0,754) | Tắt LF | 0,649 | [-0,173; -0,026] | 63,8% | 6,0% |
| T (0,759) | Bỏ template | 0,759 | [-0,002; +0,001] | 1,7% | 4,7% |
| T | Template không liên quan | 0,759 | [-0,001; 0] | 3,1% | 0,7% |
| T | Che token template | 0,758 | [-0,002; 0] | 0,2% | 0% |
| L (0,770) | Random hóa module Lens | 0,770 | Lệch logit đúng 0 | 0% | 0% |
| L | Thay LoRA bằng của B | 0,759 | Không tính | Đổi fracture 2,5% | Không đo |
| VLPT (0,774) | Tắt V | 0,772 | [-0,009; +0,001] | 0% | 1,7% |
| VLPT | Tắt P | 0,708 | [-0,118; -0,003] | 23,3% | 11,7% |
| VLPT | Bỏ T | 0,775 | [-0,002; +0,007] | 4,0% | 10,0% |

**Cách đọc nhánh P:** LF mang thông tin riêng của từng mẫu, vì tráo là hỏng. Nhưng ở P, khi bỏ LF thì đường tắt θ, β vẫn cho AP gần như cũ, nên thông tin đó trùng với thứ đã có. Ở PT và VLPT, bỏ LF làm giảm AP rõ hơn, nên không thể khái quát cho mọi cấu hình. Thao tác này đưa mô hình ra khỏi phân phối đã huấn luyện, nên chỉ nên đọc ở mức định tính.

**Mô hình có dùng ảnh và câu hỏi không: có, ở mọi nhánh v5.** Tráo ảnh đưa AP về 0,19–0,22 (gần tỷ lệ dương 0,178). Tráo câu hỏi đổi 51–65% đáp án closed và 90–93% token đầu của LLM.

**v1–v3, 300 ảnh component-clean test.** Đây là phép đo độ nhạy, không phải ước lượng hiệu năng. v1 dùng box hình chữ nhật từ `detector_clean` thay cho ngân hàng mask ROI gốc.

| Checkpoint | Gate | Can thiệp | AP | Lệch logit | Đổi closed | Đổi token đầu LLM |
|---|---:|---|---:|---:|---:|---:|
| v1 V | Không có | Tắt V | 0,720 → 0,720 | 0,000 | 0% | 2,5% |
| v1 V | Không có | Dịch box | 0,720 → 0,720 | 0,000 | 0% | 0% |
| v1 L | Không có | Tắt L | 0,745 → 0,348 | 3,094 | 9,6% | 0% |
| v1 L | Không có | Dịch box | 0,745 → 0,760 | 0,054 | 0% | 0% |
| v1 P | Không có | Tắt P | 0,533 → 0,381 | 3,296 | 62,8% | 13,2% |
| v1 VLP | Không có | Tắt V | 0,614 → 0,614 | 0,000 | 0% | 0% |
| v1 VLP | Không có | Tắt L | 0,614 → 0,311 | 1,384 | 3,6% | 0% |
| v1 VLP | Không có | Tắt P | 0,614 → 0,584 | 3,158 | 5,2% | 24,9% |
| v1 none | Không có | Bỏ T | 0,714 → 0,707 | 0,100 | 1,2% | 14,5% |
| v2 V | 0,0028 | Gate × 100 | 0,7585 → 0,7592 | 0,007 | 0,4% | 0% |
| v2 L | -0,0011 | Gate × 100 | 0,7129 → 0,7155 | 0,005 | 0% | 0% |
| v2 V | | Bỏ T | 0,7585 → 0,7559 | 0,624 | 6,0% | 5,3% |
| v3 P | 0,0005 | Gate × 100 | 0,7468 → 0,7475 | 0,026 | 0,4% | 0% |
| v3 T | Không có | Bỏ T | 0,7603 → 0,7507 | 0,310 | 37,2% | 30,5% |

Bằng chứng v3 đã lưu cho V_real (`trained_V_strength_TRAIN_only.json`): gate 0,0041, độ lệch tương đối 4,5e-5, 0/4 câu trả lời thay đổi. Checkpoint V_real và L_real của v3 không còn trong thư mục, nên không can thiệp lại được.

Ở v1, các nhánh có AP canonical thấp hơn `none` khi bật P (0,533 so với 0,714).

## 5. Tầng 3: attention và bản đồ không gian

**Attention mass (model-validation).**

| Vị trí đo | Giá trị | Mốc đều | Cách đọc |
|---|---:|---:|---|
| V, Q-Former trên token local | 0,751 | 0,750 | HAG phân bố đều, không chọn lọc |
| VLPT, Q-Former trên token local | 0,739 | 0,750 | Tương tự |
| T, fusion trên token template | 0,499 | 0,550 | Thấp hơn mức đều |
| PT, fusion trên token template | 0,296 | 0,551 | Thấp hơn mức đều |
| P, `gm.cross` entropy chuẩn hóa | 1,000 | 1,000 | Ngân hàng đáp án được chú ý đều tuyệt đối |
| P, LF trên token ảnh | 0,814 | 0,941 | Có ưu tiên token câu hỏi |
| V4, gate trung bình | 0,342 | | Đã mở từ khởi tạo 0,018 |
| V, gate trung bình | 0,002 | | Gần như đóng |
| VLPT, gate trung bình | 0,041 | | Gần như đóng |
| V4, delta so với global | 20,0% | | Đáng kể |
| V, delta so với global | 0,035% | | Không đáng kể |
| VLPT, delta so với global | 1,1% | | Không đáng kể |

**Crop của V4 (ảnh gãy có box).** Attention gần như không ưu tiên crop trúng tổn thương:

| Tập | Có crop trúng (IoU ≥ 0,1) | Crop chú ý nhất trúng | Crop ngẫu nhiên trúng |
|---|---:|---:|---:|
| Model-validation (49 ảnh có đề xuất) | 83,7% | 75,5% | 76,9% |
| Test (121 ảnh có đề xuất) | 84,3% | 75,2% | 74,5% |

**Grad-CAM và rollout (57 ảnh gãy model-validation).** Lift là tỷ lệ khối lượng bản đồ trong box chia cho tỷ lệ diện tích; 1 là ngẫu nhiên. Đối chứng dùng box của một ảnh gãy khác.

| Nhánh | CAM lift | Đối chứng | CAM đỉnh trúng box | Rollout lift | Đối chứng |
|---|---:|---:|---:|---:|---:|
| B | 0,58 | 0,39 | 1,8% | 1,10 | 1,01 |
| T | 0,31 | 0,26 | 1,8% | 1,10 | 1,01 |
| P | 0,27 | 1,13 | 0% | 1,10 | 1,01 |
| PT | 2,50 | 0,52 | 8,8% | 1,10 | 1,01 |
| L | 1,28 | 0,36 | 5,3% | 1,11 | 1,02 |
| V | 0,39 | 0,33 | 5,3% | 1,10 | 1,01 |
| VLPT | 0,98 | 0,38 | 3,5% | 1,10 | 1,01 |
| V4 | 0,59 | 0,72 | 10,5% | 1,10 | 1,01 |
| V4, test (n = 143) | 0,98 | 0,94 | 10,5% | 1,13 | 1,03 |
| Placebo V4 | 0,61 | 0,73 | 10,5% | 1,10 | 1,01 |

Ở V4, crop chiếm khoảng 22–24% attribution, còn lại ở ảnh toàn cục. Không nhánh nào cho thấy tập trung nhất quán vào tổn thương: CAM dao động mạnh và rollout hầu như không tách khỏi đối chứng. Các số này mang tính mô tả, không có CI. Không nên dùng để minh họa rằng mô hình nhìn đúng tổn thương.

## 6. Tầng 4: ablation và thống kê

**Leave-one-out đã huấn luyện chỉ có ở v1.** Lưới V/L/P của v1 chạy trên test gốc có rò rỉ trùng lặp, và có nhóm dùng box bác sĩ:

| Nguồn | Hiệu ứng | Kết quả |
|---|---|---|
| Lưới v1, main effect overall | P | -3,2 đến -4,1 điểm % ở cả ba chế độ |
| Lưới v1, main effect overall | V | Tự động +0,30 [-0,07; +0,66]; detector +1,55 [+1,13; +1,98]; bác sĩ -0,06 [-0,45; +0,32] |
| Lưới v1, main effect overall | L | Tự động -0,33 [-0,70; +0,05]; detector -0,56 [-0,98; -0,13]; bác sĩ +2,48 [+1,77; +3,19] |
| EXP4, 5 seed, VLP trừ none | Tự động | -1,53 điểm % [-2,81; -0,36] |
| EXP4, 5 seed, VLP trừ none | Detector | -1,59 điểm % [-2,90; -0,52] |
| EXP4, 5 seed, VLP trừ none | Bác sĩ | +1,27 điểm % [-0,67; +3,32] |
| Mandatory clean, 5 seed, V trừ none | Overall | -5,71 điểm % [-11,91; +0,46] |

**v5 không có leave-one-out đã huấn luyện**, vì các ô VLP, VLT, VPT, LPT chưa chạy. Phần thay thế tốt nhất là tắt từng nhánh trên VLPT khi suy luận (mục 4), nhưng thao tác này không tương đương huấn luyện lại.

**Tích lũy:** chỉ có chuỗi B → T → PT và B → P → PT. Tương tác P × T về AP trên test là +0,005 [-0,007; +0,019].

**Theo loại câu hỏi.** V4 so với B, 5 fold, test, bootstrap theo ảnh:

| Nhóm | n | B | V4 | Chênh lệch, CI 95% |
|---|---:|---:|---:|---|
| Có gãy xương (canonical) | 4.013 | 87,7% | 89,9% | +2,2 [+1,4; +3,1] |
| Nhiều phim trong ảnh | 412 | 91,3% | 94,2% | +2,9 [+1,0; +5,1] |
| Bất thường (open) | 946 | 89,5% | 90,7% | +1,2 [-0,3; +2,6] |
| Đếm vị trí gãy (open) | 972 | 87,4% | 88,3% | +0,8 [-0,6; +2,3] |
| Mô tả (open) | 2.457 | 80,4% | 80,8% | +0,4 [-0,5; +1,4] |
| Vùng có gãy không | 489 | 87,7% | 87,9% | +0,2 [-1,4; +1,6] |
| Vị trí gãy (open) | 158 | 50,0% | 50,6% | +0,6 [-1,3; +2,5] |
| Vùng cơ thể (open) | 2.534 | 95,7% | 95,6% | -0,1 [-0,6; +0,4] |
| Góc chụp | 2.371 | 92,2% | 92,2% | +0,1 [-0,4; +0,5] |
| Vùng cơ thể (yes/no) | 2.303 | 94,6% | 94,6% | 0,0 [-0,4; +0,3] |

Lợi ích của V4 tập trung ở câu hỏi gãy xương, và không lan sang câu hỏi "vùng X có gãy không" hay "gãy ở đâu". Điều này cũng khớp với cơ chế có/không có đề xuất.

Nhóm giảm rõ ở fold 0 test, không hiệu chỉnh đa so sánh:
- **P:** nhiều phim -13,6 [-21,0; -6,2]; góc chụp -2,3 [-3,8; -0,9]; vùng -1,1 [-2,1; -0,2].
- **VLPT:** nhiều phim -11,1 [-18,5; -4,9]; góc chụp -2,5 [-4,1; -0,9]; vùng -1,1 [-2,1; -0,2].
- **T:** góc chụp -2,0 [-3,4; -0,9]; vùng -0,8 [-1,7; -0,2].
- **V và L:** không nhóm nào có CI loại trừ 0.
- **V4 thật so với placebo:** nhiều phim -8,6 [-14,8; -2,5], phần còn lại CI chứa 0.

**Ngưỡng định tuyến.** Trong 1.432 hàng model-validation, 779 được luật câu hỏi xếp closed. Trong 653 hàng còn lại, type head rất phân cực: thay ngưỡng từ 0,1 đến 0,99 không đổi hàng nào ở B và V4, và đổi 3 hàng ở VLPT tại 0,99. Router không phải điểm nhạy.

**Gate local của V4 (nhân delta, fold 0 test):**

| Hệ số | 0 | 0,25 | 0,5 | 1 | 2 | 4 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AP | 0,629 | 0,681 | 0,712 | 0,741 | 0,746 | 0,728 | 0,500 |
| Sensitivity | 28,7% | 39,9% | 44,8% | 65,0% | 79,0% | 86,7% | 88,1% |
| Specificity | 98,8% | 98,0% | 96,5% | 93,8% | 84,5% | 76,1% | 68,6% |

Đáp ứng theo liều rõ ràng, nên đường này thật sự là núm điều khiển độ nhạy.

## 7. Đề xuất cho bài báo

1. Giữ V4 > B như kết quả chính, gọi là "detector-informed local route".
2. Thêm đối chứng B + tín hiệu detector (mục 2) và báo cáo cùng CI. Người phản biện có thể tự tìm ra điểm này từ placebo.
3. Bỏ mọi câu ngụ ý mô hình dùng đúng vị trí tổn thương. Câu có thể viết: "The gain was largely reproduced by late fusion of the baseline score with proposal availability and count, so it should not be attributed to lesion-specific visual evidence."
4. Báo cáo V, L, P, T là không có lợi ích AP trong cấu hình này, kèm lý do cơ chế: V có gate đóng và HAG chú ý đều; L không có đường suy luận; PF không có tác dụng; T có attention dưới mức đều.
5. Không dùng Grad-CAM làm minh họa định vị.
6. Sửa câu "so local information enters at a small scale" trong bản nháp: đó chỉ là lúc khởi tạo, gate đã học là 0,32–0,34.

## 8. Giới hạn của lần rà soát

- Chỉ một seed. Không có ước lượng nhiễu giữa các seed cho từng nhánh v5, nên số câu trả lời thay đổi so với B không tách được khỏi nhiễu huấn luyện.
- Can thiệp sau huấn luyện khác với huấn luyện lại không có nhánh đó. Tắt nhánh có thể đưa mô hình ra khỏi phân phối đã học.
- "Nhiễu cùng phân phối" giữ nguyên trung bình và độ lệch chuẩn của token crop trong lô, nên chưa phải mất hết thông tin thống kê.
- Model-validation chỉ có 57 ảnh gãy. Chỉ V4 được lặp lại trên test.
- Vì selection đã khóa, việc đọc test ở đây không thay đổi lựa chọn nhánh, nhưng phần cơ chế và đối chứng ghép muộn là phân tích hậu kiểm, không được khóa trước.
- LLM chỉ đo ở token đầu với tối đa 60 câu open cho các nhánh v5 chạy sau khi giảm tải (T, L, V4 model-validation là 150), không sinh trọn câu trả lời theo từng can thiệp.
- Các hàng closed ngoài canonical cũng được giảm còn 150 ở các nhánh chạy sau. AP không bị ảnh hưởng, nhưng tỷ lệ đổi closed giữa các nhánh không hoàn toàn cùng tập hàng.
- v1 dùng box chữ nhật thay cho mask gốc; AP của v1 trên cohort này không phải ước lượng sạch, vì một số checkpoint huấn luyện trên split có rò rỉ.
- Cohort v1–v3 giữ ImageFile.LOAD_TRUNCATED_IMAGES = True, nên có thể gồm ảnh mà giao thức sau này loại vì lỗi giải mã. Phép đo chỉ dùng cho độ nhạy.
- Checkpoint V_real, L_real của v3 đã mất, chỉ còn số liệu đã lưu. v2 không có checkpoint P đầy đủ ngoài smoke.
