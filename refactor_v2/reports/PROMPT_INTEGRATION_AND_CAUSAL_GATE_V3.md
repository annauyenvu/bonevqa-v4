# BoneVQA v3 — Protocol tích hợp prompt và causal gate

Trạng thái: **BLOCKED FOR LONG TRAINING** cho đến khi toàn bộ gate kỹ thuật và causal bên dưới vượt qua.

Mục tiêu của protocol này không phải chứng minh rằng “bật thêm module làm Accuracy tăng”, mà phải chứng minh phần tăng đến từ **thông tin đúng mà prompt cung cấp**, không phải từ số tham số, hình vẽ màu, số lượng detector box, answer prior, khác biệt khởi tạo hoặc leakage.

## 1. Đối chiếu với phương pháp gốc

### V — FAVP

Nguồn chính:

- Paper: https://doi.org/10.1609/aaai.v39i9.33047
- Code tác giả: https://github.com/OpenMICG/FAVP

FAVP tạo mask tự động bằng SAM-Med2D, chuyển từng mask thành prompt cục bộ, mã hóa ảnh gốc và các ảnh prompt bằng **shared ViT**, nối local/global tokens, rồi dùng Q-Former + linear projection làm visual prefix cho LLM. Code tác giả thực hiện `local_feats + global_feats -> Q-Former -> LLM`; khi không có local prompt, chỉ còn đường ảnh gốc.

Quyết định cho BoneVQA v3:

- Giữ shared BiomedCLIP ViT + LoRA và Q-Former.
- V chỉ được tạo `delta = QFormer(global + local) - QFormer(global)`; không ROI thì delta phải bằng 0 chính xác. Không được tạo một nhánh global-capacity khác rồi gọi đó là V.
- Automatic inference chỉ được dùng detector/SAM proposal sinh tự động; annotation-derived ROI không được phép đi vào đường này.
- BoneVQA dùng detector box thay SAM-Med2D là **biến thể FAVP-inspired**, không được mô tả là tái lập nguyên bản.

### L — Localization Lens

Nguồn chính:

- Paper: https://doi.org/10.1007/978-3-032-05114-1_33
- Code tác giả: https://github.com/CVLABLUMS/localizationlens

Phương pháp gốc tạo bốn biểu diễn: ảnh gốc, single-color segmentation, multi-color segmentation và masked image. Khi train, ảnh gốc đi cùng hai lens được chọn ngẫu nhiên, các biểu diễn được mã hóa độc lập, fusion bằng attention, giảm token bằng pixel shuffle/unshuffle, và alignment hai pha bằng DCL rồi InfoNCE. Bài báo nêu rõ inference có thể dùng **ảnh gốc duy nhất**.

Quyết định cho BoneVQA v3:

- Lens là training/alignment augmentation; production inference mặc định nhận ảnh gốc duy nhất.
- Nếu vẫn dùng LensFusion ở inference, đó là một biến thể riêng và phải có ablation original-only tương ứng; không được gộp với claim của paper.
- Real lens phải được so với nhánh có cùng tham số nhưng chỉ thấy duplicated/original views và với shuffled/matched ROI.
- Phải báo riêng localization-question metrics, không chỉ Overall Accuracy.

### P — LaPA

Nguồn chính:

- Paper: https://arxiv.org/abs/2404.13039
- Code tác giả: https://github.com/GaryGuTC/LaPA_model

LaPA gồm: latent prompt tương tác với toàn bộ answer-token bank; consistency loss kéo prompt về embedding đáp án đích chỉ trong training; latent prompt lần lượt lọc language, image và multimodal features; prior-fusion dùng GNN trên đồ thị organ–disease; cuối cùng kết hợp latent information với hai cross-modal streams.

Quyết định cho BoneVQA v3:

- Consistency target phải là embedding của **ground-truth answer trong training**, không phải image feature.
- Answer bank chỉ được xây từ train vocabulary và phải tái tạo deterministically sau load; không được ghi đè trọng số prior đã học.
- Danh sách label đơn giản không phải organ–disease graph. Nếu chưa có graph + GNN, module chỉ được gọi là **LaPA-inspired GM+LF**, không được claim Prior-Fusion của LaPA.
- Thứ tự fusion chuẩn là language → image → multimodal. Mọi residual gate phải được báo giá trị và contribution norm; gate gần 0 nghĩa là P không được model sử dụng.

### T — textual prompt

T hiện chỉ là instruction template, không phải bằng chứng về clinical grounding. Baseline thật phải là raw question, `use_textual_prompt=false`. T chỉ được giữ nếu vượt raw question, length/token-matched irrelevant instruction và vẫn ổn định qua paraphrase. Region text chỉ được dùng nếu vùng đó có sẵn hợp lệ ở production; annotation-derived region name thuộc assisted/control mode.

## 2. Universal technical gate

Một module chỉ được train dài khi tất cả điều kiện sau pass trên unit test và real-data smoke:

1. Gradient hữu hạn, khác 0 ở mọi tensor trainable dự kiến.
2. Tensor thực sự thay đổi sau optimizer step và sau resume; baseline snapshot phải lấy **sau** khi nạp resume.
3. Save/load round-trip bảo toàn output trong tolerance đã khóa.
4. Các run matched-seed có shared initialization byte-identical; module tùy chọn không được làm lệch RNG của shared heads/fusion.
5. Input/proposal/checkpoint/config đều có hash và provenance; không có fallback model hoặc download trong sealed run.
6. Với V/L/P, residual norm hữu hạn, khác 0 khi có signal, bằng 0 khi signal bị tắt, và thay đổi theo input.
7. Config resolver phải xác nhận tên run đúng với module thực tế; `None`/B không được ngầm bật T.

## 3. Causal/placebo gate

Mỗi so sánh dùng cùng split, seed, optimizer, số update, checkpoint rule và parameter budget.

### V

- B: ảnh + raw question.
- V-real: detector ROI đúng.
- V-no-ROI-capacity: cùng architecture/parameters nhưng visual delta bị khóa 0.
- V-shuffled: ROI từ ảnh donor, hoán vị ở image level.
- V-matched-shift: cùng số box và diện tích, đặt ở vị trí không-lesion.
- V-geometry-only: tọa độ/count/area + question, không pixel.
- Inference interventions: original, lesion-mask, matched non-lesion mask, background replacement.

V chỉ pass semantic gate nếu V-real vượt matched controls và thay đổi prediction/probability mạnh hơn khi can thiệp lesion so với vùng non-lesion. Gain chỉ bằng geometry-only được ghi nhận là detector shortcut, không phải visual grounding.

### L

- L-real lenses trong training; ảnh gốc duy nhất ở inference.
- L-original-only placebo: cùng LensFusion/parameter budget nhưng mọi view là ảnh gốc.
- L-shuffled/matched ROI.
- DCL/InfoNCE và pixel-(un)shuffle phải được ablate riêng nếu được claim.

L chỉ pass nếu lợi ích còn tồn tại ở original-only inference và vượt cùng kiến trúc không có semantic lens.

### P

- P-real train-answer bank.
- P-bank-off/unconditioned với cùng số latent token và tham số.
- P-random/irrelevant bank có cùng shape/token count.
- P-shuffled-context nếu dùng contextual variant.
- P-target-consistency on/off.

P chỉ pass nếu answer bank + target consistency vượt controls, latent residual có input dependence, và checkpoint round-trip không đổi prediction.

### T

- raw question.
- clinical-neutral instruction.
- length/token-matched irrelevant instruction.
- semantic-preserving paraphrases.

T chỉ pass nếu tăng ổn định, không phụ thuộc đúng một câu template, và không chứa label/region không có ở inference.

## 4. Clinical/statistical gate

- Screening một seed chỉ quyết định có đáng chạy tiếp, không dùng làm claim.
- Module phải không làm giảm endpoint chính quá non-inferiority margin đã khóa.
- Fracture metrics chỉ dùng canonical image-level fracture-presence question, một record/image.
- Báo AUROC, AUPRC, sensitivity, specificity, balanced accuracy, Brier, ECE và image-clustered bootstrap CI.
- Calibration fit trên calibration cohort độc lập; Platt parameters và threshold phải được áp dụng thật ở runtime.
- Multi-seed chỉ chạy cho B và module/combo đã vượt technical + causal + clinical gate.

## 5. Quy tắc chọn Final Solution

Final Solution là cấu hình nhỏ nhất trong số các cấu hình pass toàn bộ gate. Một module bị loại nếu:

- gain không vượt matched placebo/capacity control;
- residual/gate cho thấy module bị tắt về mặt chức năng;
- cần annotation ROI ở automatic inference;
- chỉ khai thác proposal geometry/answer prior/template artifact;
- hoặc làm giảm endpoint lâm sàng vượt margin.

Vì vậy việc Final Solution tắt một hay nhiều module không phủ nhận thí nghiệm; nó là kết quả đúng nếu module đó không chứng minh được giá trị nhân quả trên pipeline production-parity.

## 6. Blocker đã xác nhận trước retraining

- `None` cũ thực tế bật T.
- Fracture selector cũ trộn global presence với region/wrong-region questions.
- Detector ROI control cũ bị bỏ qua khi `boxes_from` được dùng.
- Proposal existence/count/area tự nó dự đoán fracture khá mạnh; phải có geometry-only control.
- V/L vẫn tạo nhánh global có capacity khi không có ROI.
- P refresh sau load có thể ghi đè trained prior và answer bank khác nhau giữa train/eval do dropout.
- Conditional modules làm lệch RNG của shared randomly initialized heads giữa các ablation.
- L cũ dùng lens ở inference, trái với protocol original-image-only của paper.
- 25 ảnh có EXIF orientation khác 1 (trong đó 2 ảnh fracture có box); detector cũ đọc raw orientation trong khi VQA loader canonicalize ảnh.
- Test đã được xem trong quá trình sửa protocol; kết quả tiếp theo là development/pilot cho đến khi có external untouched holdout hoặc nested CV được khóa.

## 7. Trạng thái triển khai sau audit paper/code (15-10-2026)

Đã sửa và có unit test:

- Baseline v3 `B` là ảnh + raw question, tắt cả V/L/P/T; config resolver fail-closed nếu tên variant không khớp module thực tế.
- Fracture endpoint dùng một predicate allow-list duy nhất trong train, loss, inference, metric và calibration. Câu region/multiplicity không còn bị trộn vào endpoint toàn ảnh.
- V dùng `QFormer(global + local) - QFormer(global)`, Q-Former dropout bằng 0 trong paired delta, projection bias-free và chỉ route một lần qua shared fusion. Không ROI hoặc gate-zero cho delta bằng 0 chính xác.
- L không đi trực tiếp vào answer endpoint trong protocol chuẩn. LensFusion là teacher/alignment path ở train và stage-1 validation; answer loss ở train/test đều dùng original-image stream. Stage 1 dùng DCL, stage VQA dùng InfoNCE. Các control real/original-only/shuffled/matched phải được pretrain và fine-tune riêng.
- P tạo latent prompt từ train-answer bank; fusion theo language → image → multimodal; consistency target là ground-truth answer embedding chỉ trong train. Answer bank được encode deterministically; refresh sau load không còn ghi đè prior. Nếu không có graph+GNN thì `prior_labels=[]` và tên claim chỉ là LaPA-inspired GM+LF.
- T dùng cùng chuỗi câu hỏi ở biomedical text encoder và LLM, không còn nested `Question:`. Có raw, irrelevant-length-like và clinical paraphrase configs.
- `boxes_from` hiện thực sự chịu tác động của shuffled/generic/matched/matched-all controls. `matched_all` tạo placebo geometry-matched cho cả detector-positive và detector-negative. Direct inference với detector box vẫn rasterize được proposal khi SAM bị tắt.
- Optional modules khởi tạo trong RNG scope cô lập; module seed thay đổi theo experimental seed nhưng giống nhau giữa các variant cùng seed.
- Inference ablation dùng gate-zero flags, không đổi architecture rồi cố load checkpoint khác hình dạng.
- Runtime calibration áp dụng thật Platt slope/intercept rồi mới dùng ngưỡng lâm sàng; artifact bắt buộc chứng minh test không tham gia fit/chọn ngưỡng và được ghi SHA-256.
- Detector train/export dùng chung EXIF-canonical coordinate contract. Checkpoint/proposal/split manifest được hash; artifact detector v2 cũ hiện bị fail-closed và không thể dùng cho v3.
- Production app/CLI tự chạy detector v3 khi V được bật và không có box bác sĩ; app ghi detector/calibration provenance. Annotation ROI không được ngầm dùng ở automatic inference.
- Bộ unit test hiện tại: 58/58 pass; 23/23 config v3 resolve và validate thành công.

Smoke thực trên component-clean data với full BiomedCLIP + Qwen runtime:

- B, V, P, T đều hoàn thành một lượt smoke giới hạn; Visual-LoRA gradient/update pass 24/24 tensor.
- V có gradient và update thật ở `gamma_visual`; mean semantic contribution sau các optimizer step là `1.03e-5` (nhỏ như kỳ vọng từ zero-init gate, chưa phải bằng chứng hiệu quả).
- P có consistency loss khác 0, latent contribution khác 0 và các tensor latent/fusion nhận gradient/update.
- P checkpoint được load hai lần độc lập: sai khác tối đa bằng 0 cho answer bank, `x_lp`, latent delta, fused representation, closed logits, fracture logit và toàn bộ P state.
- L smoke lần đầu phát hiện batch detector-negative không có supervision khả vi. Training loop nay ghi/skip đúng batch này thay vì crash hoặc tạo surrogate ROI; lượt sửa lại pass gradient/update LensFusion. Đây là technical wiring gate, chưa phải claim L tốt hơn B.

## 8. Cập nhật sau detector v3 và production smoke (15-10-2026)

Detector EXIF-canonical v3 đã hoàn tất bằng early stopping ở epoch 11/12. Checkpoint `refactor_v2/detector_clean_v3/best.pt` và báo cáo xác nhận `preprocessing_version=bonevqa-exif-canonical-v3`. Kết quả clean test gồm 550 ảnh (119 dương, 431 âm): mean best-IoU 0.2530, IoU≥0.5 17.65%, miss rate 31.09%, specificity trên ảnh âm 69.37% (FPR 30.63%). Đây là chất lượng proposal còn yếu; V/L chưa được coi là có grounding lâm sàng. Historical 108 ảnh đều dương tính: mean IoU 0.2804, IoU≥0.5 23.15%, miss 30.56%; cohort này chỉ dùng để so sánh lịch sử, không thay clean test.

Proposal export v3 đã hoàn tất và được kiểm tra: 4,083/4,083 ảnh, train/val/test lần lượt 2,890/643/550; SHA-256 của checkpoint, split manifest và proposal JSON đều khớp sidecar provenance; tọa độ khai báo theo EXIF-canonical pixels. Artifact proposal v2 không được dùng.

Geometry-only control đã chạy trên các split phát triển. Trên test đã xem, fracture AUROC 0.8185, AUPRC 0.6369, BA@0.5 0.7800 và overall accuracy 0.6234; vì control không đọc pixel ảnh, đây là bằng chứng mạnh rằng câu hỏi/template và hình học proposal có thể giải thích một phần lớn kết quả. Các số này là audit phát triển, không phải đánh giá độc lập.

Loader-level kiểm tra với proposal v3 xác nhận control hoạt động ở cả hai nhóm: proposal-positive giữ hình học qua `matched`; `matched_all` tạo ROI placebo cho ảnh detector-negative (ví dụ âm chuyển 0 box → 2 box); `shuffled` lấy ROI từ donor khác ở mức image. Đây mới là xác nhận phép biến đổi, chưa phải so sánh nhân quả về accuracy; để có so sánh đó cần huấn luyện các cấu hình V/L tương ứng cùng seed và ngân sách.

Production detector→VQA smoke đã gọi detector v3 tự động khi V bật và không truyền box thủ công, trả về 3 proposal cùng prompt/lens images. Lượt đầu phát hiện `predict.py` khởi tạo `torch.autocast(device_type="mps")` không được hỗ trợ trên MPS ngay cả khi `enabled=False`; đã sửa để chỉ autocast CUDA, dùng context rỗng trên CPU/MPS. Smoke chạy lại hoàn tất. Checkpoint dùng trong smoke là checkpoint smoke nhỏ, nên chỉ chứng minh wiring/runtime, không phải chất lượng VQA.

Pixel interventions (original/lesion-mask/matched-non-lesion/background replacement) chưa được dùng làm kết quả semantic: hiện chưa có VQA checkpoint được huấn luyện bằng proposal v3 và vượt gate. Có thể chạy smoke kỹ thuật trên checkpoint nhỏ, nhưng kết quả đó không đủ để kết luận model dùng lesion pixels. Tương tự, chưa có trained VQA comparison cho matched/matched-all controls.

Pixel-intervention technical smoke đã hoàn tất trên historical cohort 108 ảnh (499 QA) với checkpoint `smoke_real_v_prompt_v3` và proposal v3. Bốn điều kiện chạy hoàn tất với detector proposal cố định: lesion-mask tạo flip rate 22.44% và mean |Δp_yes| 0.00320; matched-non-lesion tương ứng 16.03% và 0.00126; background replacement 49.10% và 0.01636. Điều này xác minh implementation phản ứng khác nhau với lesion và matched control theo expected direction. Tuy nhiên checkpoint smoke có overall accuracy 4.21%, fracture sensitivity 0%, và mọi condition có accuracy change 0.00 pp. Các số này chỉ là wiring audit, không được dùng làm evidence về grounding, hiệu quả hay endpoint lâm sàng.

Trước one-seed screening vẫn cần: (1) khóa external holdout hoặc honest nested-CV vì clean test đã được xem; (2) chạy pixel interventions và các matched-placebo so sánh trên checkpoint đủ điều kiện; (3) ghi rõ threshold/proposal score đã khóa trước đánh giá. Chưa chạy long VQA training cho đến khi các control/gate này được seal; detector v3 hiện là đầu vào hợp lệ để bắt đầu các smoke/controls tiếp theo, không tự nó chứng minh V hay L có ích.

Do đó trạng thái vẫn là **HARD STOP FOR LONG TRAINING**.

