# BoneVQA v3 và nghiên cứu thực nghiệm về prompt trong hỏi đáp ảnh y khoa

Báo cáo nghiên cứu tổng hợp • Bản xuất ngày 27 tháng 10 năm 2026

## Tóm tắt

Hỏi đáp ảnh y khoa đòi hỏi mô hình vừa hiểu câu hỏi, vừa sử dụng đúng bằng chứng trên ảnh. Đối với X-quang xương, một hệ thống có thể trả lời đúng nhiều câu hỏi về phương thức chụp hoặc vùng cơ thể nhưng vẫn bỏ sót gãy xương. Vì vậy, nghiên cứu BoneVQA v3 xem độ chính xác hỏi đáp và năng lực phát hiện gãy là hai nhóm kết quả riêng biệt. Hệ thống kết hợp BiomedCLIP, PubMedBERT, hợp nhất đa phương thức, QFormerLite, tiền tố thị giác và Qwen2.5-0.5B-Instruct; đồng thời khảo sát visual prompt, Lens, latent prompt và textual prompt bằng các đối chứng phù hợp.

Đánh giá baseline B theo năm fold trên 4.013 ảnh, 3.920 nhóm liên quan đã biết và 17.998 cặp hỏi đáp đạt exact accuracy 90,00%. Tuy nhiên, sensitivity phát hiện gãy chỉ đạt 55,94%, specificity 94,57% và balanced accuracy 75,26%, với khoảng bootstrap 95% là 73,16–77,05%. Hiệu chuẩn xác suất cải thiện Brier/ECE nhưng làm balanced accuracy giảm 1,74 điểm phần trăm. Các thí nghiệm prompt chưa xác lập một thành phần cải thiện ổn định so với đối chứng tương ứng. Ba pilot tiếp theo sử dụng tài nguyên gốc LaPA, FAVP và Lens đã hoàn tất trong ngân sách giới hạn: LaPA/PF không cải thiện tương phản chính; FAVP và Lens đạt 6/8 câu đúng nhưng đều dự đoán lớp “no”, bằng baseline đa số.

Đóng góp được chứng minh của công trình là một hệ thống thực nghiệm có truy vết dữ liệu, kiểm tra gradient, tái lập từ checkpoint và phân tích đối chứng; chưa phải một kiến trúc đạt ưu thế lâm sàng. Dữ liệu đánh giá đã tham gia quá trình phát triển, nhóm component chưa được xác nhận là bệnh nhân và các pilot có nhiều điều chỉnh so với công trình gốc. Báo cáo giữ nguyên kết quả âm, trình bày đầy đủ các giới hạn và xác định những thử nghiệm còn thiếu trước khi có thể kết luận về grounding thị giác hoặc triển khai thực tế.

Từ khóa: medical visual question answering; X-quang xương; visual prompting; parameter-efficient fine-tuning; localization; reproducibility; negative controls.

## Phạm vi và quy tắc đọc báo cáo

Đây là báo cáo tổng hợp bằng tiếng Việt theo cấu trúc nghiên cứu quốc tế, không phải tuyên bố một bài báo đã được bình duyệt. Nội dung dựa trên mã nguồn, cấu hình chạy, báo cáo kết quả và các artifact đã khóa trong dự án. Mã triển khai hiện hành nằm trong thư mục `refactor_v2`; tên thư mục không có nghĩa toàn bộ kết quả được gọi là “v2”. “v3” chỉ giai đoạn giao thức và refinement của nghiên cứu.

Ba tên cần được phân biệt xuyên suốt: **B** là baseline tắt T/V/L/P; **BoneVQA mở rộng** là các biến thể bật từng cơ chế và đối chứng; **LaPA, FAVP, Lens dùng tài nguyên gốc** là ba nghiên cứu riêng có checkpoint và giao thức riêng. Mũi tên LaPA → FAVP → Lens biểu diễn thứ tự thực hiện, không phải một mạng đã ghép nối ba phương pháp.

Các số liệu dưới đây được giữ riêng theo cohort, endpoint và ngân sách. “Hoàn tất” chỉ có nghĩa một thử nghiệm đã đạt phạm vi thực thi và kiểm toán đã khai báo. Nó không đồng nghĩa giả thuyết khoa học đúng. Các thiết kế chưa chạy được ghi rõ; không bổ sung số liệu giả để làm đầy ma trận ablation. Phụ lục dẫn đến các bảng chi tiết, nhật ký lỗi và chỉ mục báo cáo nguồn, giúp đọc lại toàn bộ hồ sơ thay vì chỉ nhìn các điểm số thuận lợi.

# 1 Tổng quan bài toán và các công trình nền

## 1.1 Bài toán Med VQA

Medical visual question answering nhận ảnh y khoa I và câu hỏi q, tạo câu trả lời a. Mục tiêu học có thể viết là tối đa hóa p(a | I, q). Câu trả lời đóng thuộc một tập hữu hạn, chẳng hạn “yes/no” hoặc tên vùng cơ thể; câu trả lời mở là một chuỗi token. Hai dạng đầu ra có thể cùng xuất hiện trong một dataset nhưng cần cơ chế đánh giá riêng.

Trong BoneVQA, các câu hỏi gồm sự hiện diện của gãy xương, vùng cơ thể, hướng chụp, số vị trí gãy, dụng cụ chỉnh hình và mô tả phát hiện. Câu hỏi canonical “Is there a fracture in this X-ray?” được dùng làm endpoint nhị phân nhất quán theo ảnh. Đây là cách tránh đếm nhiều câu hỏi gần giống nhau của một ảnh như nhiều quan sát chẩn đoán độc lập.

Đầu vào public của baseline là ảnh gốc và câu hỏi thuộc phạm vi dữ liệu hỗ trợ. Đáp án, loại câu hỏi, vùng tổn thương và annotation chỉ thuộc dữ liệu huấn luyện hoặc đánh giá. Nếu một biến thể dùng proposal tự động thì proposal phải được tạo bởi pipeline có provenance rõ ràng. Một bounding box từ annotation cung cấp thông tin mà người dùng thông thường không có, nên không thể đưa vào inference rồi gọi đó là kết quả tự động.

## 1.2 Khó khăn cốt lõi

Gãy nhỏ có thể chiếm rất ít pixel trong ảnh lớn. Giảm toàn ảnh xuống 224 × 224 giúp tiết kiệm chi phí nhưng có nguy cơ làm mất dấu hiệu tinh tế. Crop cục bộ có thể khôi phục độ phân giải, song cũng có thể bỏ ngữ cảnh, chọn nhầm vùng hoặc tạo dấu hiệu giả từ đường viền prompt. Do đó, tăng số vùng không tự động tăng khả năng chẩn đoán.

Dữ liệu hỏi đáp thường không cân bằng. Một tập có nhiều câu hỏi dễ hoặc nhiều nhãn âm cho phép mô hình đạt accuracy cao bằng prior ngôn ngữ. Câu trả lời đúng chưa chứng minh mô hình đã nhìn đúng vùng. Ngược lại, thay ảnh làm logits thay đổi chỉ xác nhận đường tín hiệu thị giác đang hoạt động, chưa chứng minh thay đổi đó có ích về ngữ nghĩa.

Rò rỉ dữ liệu có thể xảy ra ở nhiều cấp: byte ảnh trùng, ảnh biến đổi, nhiều QA của cùng ảnh, nhiều ảnh trong cùng bài báo và checkpoint upstream không rõ manifest. Việc tách theo QA đơn lẻ thường không đủ. Ngoài ra, dữ liệu pretraining của foundation model không hoàn toàn kiểm soát được; lọc dữ liệu cục bộ không xóa được mọi khả năng overlap đã có từ trước.

Khó khăn thực thi gồm giới hạn RAM, CPU chậm với một số phép FP16, sai khác phiên bản thư viện, checkpoint thiếu thông tin và tài nguyên gốc hết hạn. Nghiên cứu cần phân biệt sửa lỗi kỹ thuật với thay đổi phương pháp: sửa routing để loss đến đúng module không phải là bằng chứng module đó giúp tăng chất lượng.

## 1.3 LaPA

LaPA nghiên cứu latent prompt cho Med-VQA, gồm sinh prompt tiềm ẩn, kết hợp tri thức và fusion. Ý tưởng là đưa thông tin bổ trợ vào không gian đặc trưng thay vì chỉ thêm câu hướng dẫn bằng ngôn ngữ. Công trình được công bố tại CVPR Workshops 2024. Trong dự án, GM/LF và PF được kiểm tra theo các giai đoạn riêng; nguồn graph gốc và thứ tự node là điều kiện quan trọng để tái lập PF. [R1]

Ở bản port ban đầu, BiomedCLIP/PubMedBERT được dùng chung với BoneVQA. Ở pilot sau, các encoder và khối đa phương thức M3AE được nạp từ tài nguyên gốc, nhưng encoder ảnh/văn bản được đóng băng và graph 409 node được tái dựng từ SLAKE. Thiếu graph 577 node gốc vẫn là một khác biệt thực chất. Số 577 image tokens của ảnh 384 pixel không phải bằng chứng đã có graph 577 node.

## 1.4 FAVP

FAVP, công bố tại AAAI 2025, dùng visual prompt thích nghi để đưa thông tin vùng vào hỏi đáp sinh câu trả lời. Mục tiêu của adaptive visual prompt creator là làm rõ chi tiết cục bộ; bộ sinh câu trả lời kết hợp biểu diễn nhiều mức với fine-tuning tiết kiệm tham số. [R2]

Trong triển khai tham chiếu của dự án, các thành phần thật gồm SAM-Med2D, EVA, QFormer và Vicuna V0 7B. Trọng số nền đã tải không đồng nghĩa đã có checkpoint nối stage 1–2 của tác giả. Projection từ visual queries sang Vicuna vẫn phải khởi tạo mới; bởi vậy kết quả thuộc một adaptation từ tài nguyên gốc, không phải tái lập đầy đủ FAVP đã pretrain.

## 1.5 Localization Lens

Localization Lens, công bố tại MICCAI 2025, sử dụng các view được biến đổi từ vùng phân đoạn để cải thiện biểu diễn vị trí và quan hệ ảnh–văn bản. Công trình kết hợp biểu diễn nhiều view, nén đặc trưng và huấn luyện alignment. [R3]

Dự án phân biệt LensFusion tự xây trong v3 với pipeline nguồn Lens dùng SAMHuge và SmolVLM. Các khối có cùng tên hoặc cùng mục tiêu không mặc nhiên tương đương. Pilot nguồn sau cùng so sánh bật/tắt loss DCL/InfoNCE trên cùng ba view; nó không cô lập toàn bộ hiệu ứng tạo view và không phải ablation của mọi thành phần trong paper.

## 1.6 Câu hỏi nghiên cứu

Nghiên cứu đặt ra năm câu hỏi: baseline có học endpoint gãy và QA không; visual prompt có giúp nhờ vị trí thật hay chỉ nhờ thêm dữ liệu/compute; Lens có chuyển lợi ích định vị sang câu trả lời không; latent prompt có khai thác ngữ nghĩa bank/graph không; và các kết quả có bền vững trước đối chứng, seed và hiệu chuẩn hay không. Các thí nghiệm được tổ chức để trả lời từng câu hỏi trong phạm vi cụ thể, không dùng một điểm accuracy tổng hợp thay cho tất cả.

# 2 Cơ sở lý thuyết

## 2.1 ViT và attention

Vision Transformer chia ảnh thành patch, ánh xạ mỗi patch thành vector và thêm thông tin vị trí trước khi áp dụng attention. Với ảnh 224 × 224 và patch 16 × 16, có 196 patch; thêm CLS tạo 197 token. Self-attention cho phép trao đổi thông tin giữa các vùng, nhưng không tự đảm bảo token tập trung vào tổn thương. [R4]

Attention được biểu diễn bởi softmax(QKᵀ/√d)V. Trong self-attention, Q, K và V đến từ cùng một chuỗi; trong cross-attention, query và memory có thể đến từ hai modality khác nhau. Chi phí attention tăng theo chiều dài chuỗi, là lý do cần giới hạn số view, crop và token trước LLM.

## 2.2 BiomedCLIP và PubMedBERT

BiomedCLIP học biểu diễn ảnh–văn bản từ các cặp hình và chú thích y sinh; checkpoint dùng trong hệ thống kết hợp ViT-B/16 với PubMedBERT. Đây là một prior miền y sinh hữu ích, nhưng dữ liệu hình khoa học đa dạng không đồng nhất với bài toán gãy xương. [R5]

Trong baseline, token ảnh và token câu hỏi có chiều 768. Nhánh biểu diễn toàn cục qua projection của BiomedCLIP có thể phục vụ alignment; nó cần được phân biệt với chuỗi token dùng cho fusion. Bộ mã hóa văn bản đóng băng không có nghĩa hệ thống không học được cách dùng câu hỏi, vì các khối fusion, head và bridge phía sau vẫn được tối ưu.

## 2.3 SAM và SAM-Med2D

SAM gồm image encoder, prompt encoder và mask decoder, nhận điểm, hộp hoặc mask để sinh phân đoạn. Với automatic mask generation, lưới điểm đóng vai trò prompt tự động. Mask thể hiện vùng phân đoạn theo mô hình, không phải nhãn tổn thương được bác sĩ xác nhận. [R6]

SAM-Med2D điều chỉnh SAM cho ảnh y khoa nhằm xử lý khoảng cách miền. Tên đúng trong tài nguyên FAVP là **SAM-Med2D**, không phải SAM 2 hay Medical SAM 2. Ba tên này chỉ các hệ thống khác nhau. Các ngưỡng predicted IoU/stability lọc mask; không có mask vượt ngưỡng không tương đương ảnh không bệnh. [R7]

Trong nghiên cứu này, SAM-Med2D và SAMHuge cho kết quả coverage rất khác nhau. Sự khác biệt phải được báo cáo trước khi giải thích hiệu quả local prompt. Không được hạ ngưỡng sau khi xem validation chỉ để làm nhánh local có đầu vào.

## 2.4 LLM và Qwen

LLM tự hồi quy phân rã xác suất câu trả lời thành tích các p(aₜ | a₍<t₎, I, q). Qwen2.5-0.5B-Instruct là LLM dùng trong v3; ảnh đi vào thông qua embedding prefix học được, vì bản Qwen này tự nó không phải encoder ảnh. [R8]

Teacher forcing dùng token đáp án đúng làm ngữ cảnh khi train và chỉ tính loss trên vùng đáp án. Inference phải sinh từ câu hỏi và ảnh, không có đáp án. Greedy decoding chọn token xác suất cao nhất; giới hạn token giúp kiểm soát chi phí nhưng có thể cắt câu trả lời. Vì vậy cần lưu raw output, EOS và trạng thái chạm giới hạn thay vì chỉ lưu đáp án đã rút gọn.

## 2.5 LoRA và tiền tố liên tục

LoRA biểu diễn cập nhật ma trận bằng ΔW = (α/r)BA, với r nhỏ hơn kích thước đầy đủ. Trọng số nền được giữ cố định, còn hai ma trận hạng thấp được học. Điều này giảm số tham số cần gradient và trạng thái optimizer, nhưng không loại bỏ chi phí chạy backbone. [R9]

Prefix tuning dùng các biểu diễn liên tục để điều kiện hóa mô hình sinh. Trong v3, 32 query thị giác cộng một vector fusion được chiếu sang hidden size của Qwen và nối trước embedding câu hỏi. Đây là **visual soft prefix ở đầu vào**; không nên mô tả nó như một bản sao đầy đủ của prefix tuning key/value ở mọi tầng trong công trình gốc. [R10]

LoRA thay đổi phép biến đổi của mô hình; prefix bổ sung ngữ cảnh đầu vào. Hai cơ chế có thể cùng dùng. Tắt textual prompt T không tắt visual prefix và cũng không xóa wrapper cố định của LLM.

## 2.6 Các loại prompt trong computer vision

Prompt hình học có thể là điểm, hộp, contour hoặc vòng tròn; prompt raster có thể là vùng tô màu hay mask; prompt học được có thể là token/query hoặc latent vector. Textual prompt bổ sung mô tả bằng ngôn ngữ. Mỗi loại có một nguy cơ shortcut khác nhau: đường viền nhân tạo, màu, độ dài câu, vị trí box hoặc dung lượng bank.

Một đối chứng tốt giữ phần lớn các yếu tố đó tương đương và thay đổi yếu tố cần kiểm tra. Ví dụ, V-real so với location-placebo giữ số lượng/kích thước proposal nhưng thay vị trí; TRAIN bank so với random bank kiểm tra ngữ nghĩa trong khi cố gắng giữ dung lượng; Lens-loss so với CE-only giữ view train giống nhau. Không đối chứng nào hoàn hảo nếu compute, dropout hoặc số tham số hoạt động cũng thay đổi.

## 2.7 Các hàm mất mát

**Cross-entropy câu trả lời đóng.** Với nhãn y trong vocabulary TRAIN, L_closed = −log p(y | I,q). Nhãn ngoài vocabulary không được lặng lẽ thêm từ validation; các endpoint giới hạn vocabulary phải công bố cách tính unsupported answer.

**Cross-entropy sinh chuỗi.** L_open = −Σₜ log p(aₜ | a₍<t₎,I,q), sau đó giảm theo token hoặc theo sequence tùy recipe. Token-mean khiến chuỗi dài có trọng số lớn hơn; sequence-uniform cho mỗi đáp án trọng lượng tương đương. Hai lựa chọn đã được kiểm tra riêng, không thể gọi loss thấp hơn là tốt hơn khi mẫu số khác nhau.

**BCE nhị phân.** L_fracture = −w₊y log p − (1−y)log(1−p). Trọng số dương lấy từ TRAIN, và loss chỉ áp dụng cho endpoint được định nghĩa. Không được route mọi câu chứa từ “fracture” vào cùng nhãn hiện diện gãy.

**InfoNCE.** Với similarity s và temperature τ, loss cho cặp đúng i có dạng −log[exp(sᵢᵢ/τ)/Σⱼexp(sᵢⱼ/τ)]. Những QA của cùng ảnh không phải negative độc lập; bản v3 có logic multi-positive theo nhóm. Gradient accumulation tăng effective optimizer batch nhưng không tự tăng số negative đồng thời trong phép contrastive.

**Decoupled contrastive loss.** DCL tách positive khỏi tổng negative: dạng đơn hướng là −s_positive/τ + log Σ_negative exp(s/τ). Vì không có positive trong mẫu số như InfoNCE, giá trị loss có thể âm. Cần đủ negative thật; batch chỉ gồm một ảnh khác biệt có thể làm mục tiêu suy biến. Pilot Lens nguồn dùng batch hai ảnh thật; không sử dụng giả định accum4 tạo tám negative đồng thời.

**Consistency và tổng loss.** Các nhánh prompt có thể thêm loss consistency, alignment, answer-type hoặc spatial supervision. Tổng quát L = ΣₖλₖLₖ, nhưng λ, denominator và routing phải lấy từ config từng run. Không có một công thức trọng số duy nhất đại diện mọi thí nghiệm trong báo cáo.

## 2.8 Độ đo và ý nghĩa thống kê

| Độ đo | Định nghĩa hoặc vai trò | Điều không thể suy ra riêng từ độ đo |
|---|---|---|
| Exact match | Tỷ lệ đáp án bằng target theo normalization đã khóa | Không đo đầy đủ tương đương ngữ nghĩa của câu tự do |
| Sensitivity | TP/(TP+FN) | Không phản ánh số báo động giả |
| Specificity | TN/(TN+FP) | Cao vẫn có thể bỏ sót nhiều ca dương |
| Balanced accuracy | (Sensitivity+Specificity)/2 | Không thể thay bằng overall QA accuracy |
| AUROC | Xếp hạng trên các ngưỡng | Không xác nhận ngưỡng triển khai cụ thể |
| Average precision | Tóm tắt precision–recall theo thứ hạng | Phụ thuộc prevalence; không so tùy ý giữa cohort |
| Brier | Trung bình (p−y)² | Xác suất tốt hơn không đảm bảo sensitivity tốt hơn ở 0,5 |
| ECE10 | Sai lệch calibration theo 10 bin | Phụ thuộc bin và cỡ mẫu |
| IoU | Diện tích giao chia diện tích hợp của hai vùng | Box overlap không phải chẩn đoán hoặc segmentation Dice |
| Box-patch AP | Xếp hạng patch theo occupancy của box | Head định vị tốt không chứng minh answer head dùng nó |

Bootstrap được thực hiện theo component, dùng cùng mẫu resampling cho hai điều kiện cần so sánh. Khoảng tin cậy là bất định có điều kiện theo các model đã fit; một seed không cung cấp bất định giữa mọi seed huấn luyện. Khoảng chứa 0 không chứng minh tương đương. Khoảng [0,0] từ tám output giống nhau không chứng minh hai phương pháp tương đương trên dân số.

## 2.9 Kỹ thuật huấn luyện và tái lập

Các recipe sử dụng AdamW, gradient accumulation, gradient clipping, lịch learning rate và checkpointing tùy pha. Cần lưu số microbatch và số optimizer update riêng; hai con số không hoán đổi. Đóng băng tham số không có nghĩa có thể đặt toàn bộ forward trong no_grad nếu vẫn cần gradient đi qua mô hình đến prefix hoặc LoRA.

Fresh-process replay khởi động process mới, dựng lại mô hình và đối chiếu output từ checkpoint. Nó mạnh hơn việc gọi forward hai lần trong cùng process, nhưng chỉ xác nhận đường tái lập đã thử. Hash dependency bảo vệ tính nguyên vẹn; không phải chứng nhận đúng ngữ nghĩa hoặc không rò rỉ dữ liệu.

# 3 Dữ liệu và cách sử dụng theo từng giai đoạn

## 3.1 Nguồn dữ liệu

| Nguồn | Vai trò trong dự án | Số lượng và trạng thái được xác minh |
|---|---|---|
| ROCO cục bộ | Caption alignment upstream | 19.556 TRAIN caption; 9.903 model-validation trong run mới |
| PMC-VQA cục bộ | Generative QA alignment upstream | 28.490 TRAIN hợp lệ; 1.500 model-validation; filtered test manifest 49.990 QA, không có nghĩa đã có kết quả test hợp lệ được báo cáo ở đây |
| FracAtlas | Ảnh X-quang, nhãn/annotation gốc, QA template | 4.013 ảnh giữ lại, 3.920 component; 70 ảnh quarantine |
| VQA-RAD | Nguồn tham khảo/bổ sung lịch sử | Split QA lịch sử có ảnh chung; không dùng như test độc lập theo ảnh |
| SLAKE | Tài nguyên tham khảo và tái dựng graph PF | Graph 409 node được dùng với provenance riêng; không thay thế âm thầm graph 577 của LaPA |

Các con số trên là snapshot cục bộ đã xử lý, không phải kích thước toàn bộ dataset công bố. Đặc biệt, 100.219 đường dẫn trong byte inventory không phải 100.219 ảnh độc lập. Dữ liệu annotation gốc được giữ; không tự tạo nhãn gãy, nhãn cơ quan hay cạnh tri thức để làm thí nghiệm chạy được. [E1, E2]

## 3.2 Kiểm soát overlap và lỗi dữ liệu

Audit phát hiện `train_lon` PMC lịch sử và test có 33.252 byte-hash chung. Checkpoint stage2 lịch sử thiếu manifest không được dùng làm bằng chứng độc lập. Kết luận hợp lệ là provenance không đủ và tập lịch sử bị overlap; không được suy ra chắc chắn một checkpoint cụ thể đã train trên từng record test khi không có manifest.

Kiểm tra perceptual FracAtlas–ROCO/PMC xét 385.771.320 cặp ở pHash radius 8 và thu 87 ứng viên. Tái kiểm tra ứng viên là một bước bảo vệ bổ sung, không đảm bảo phát hiện mọi crop, rotation hay ảnh ghép. Các quan hệ bài báo PMC và quan hệ ảnh được lan truyền khi chia nhóm để tránh một đường nối gián tiếp vượt qua split.

Ba record PMC TRAIN malformed được loại, đưa 28.493 xuống 28.490; mười record malformed của filtered test được quarantine, còn 49.990. Đây là sửa tính hợp lệ schema/label, không phải đánh giá lại chẩn đoán bởi chuyên gia. FracAtlas strict decoding và nhãn mâu thuẫn dẫn tới quarantine 70 ảnh cùng các thành viên component liên quan.

## 3.3 Vai trò dữ liệu và năm fold

Mỗi fold phân biệt TRAIN, model-validation, calibration và outer. TRAIN dùng tối ưu trọng số và xây vocabulary; model-validation dùng phát triển theo giao thức; calibration chỉ fit biến đổi xác suất; outer dùng đánh giá model đã khóa. Tách theo known component, không chỉ theo QA row.

Fold 0 có 2.568 ảnh/11.502 QA TRAIN, 321 ảnh/1.432 QA model-validation, 321 ảnh/1.432 QA calibration và 803 ảnh/3.632 QA outer. Năm outer fold bao phủ mỗi ảnh đúng một lần: 4.013 ảnh/17.998 QA, gồm 715 ảnh gãy và 3.298 ảnh không gãy. Vocabulary TRAIN theo fold lần lượt là 125, 117, 115, 127 và 120; vocabulary 41 class của một pilot nhỏ không áp dụng cho toàn bộ grouped evaluation.

Dữ liệu này đã được xem trong development. Thuật ngữ “outer test” chỉ vai trò kỹ thuật trong giao thức, không biến nó thành tập xác nhận lâm sàng chưa từng được xem. Component chưa được kiểm chứng là patient ID.

## 3.4 Cohort của các nghiên cứu nhỏ

| Nhóm | TRAIN | Đánh giá chính | Lưu ý |
|---|---|---|---|
| Screening v3 cùng cohort | 256 ảnh/1.164 QA | 321 ảnh/1.432 QA | 4 epoch, 292 update; seed 42 |
| Pilot L/P lịch sử | 261 ảnh/1.173 QA | Subset 142 QA và canonical 321 ảnh | Không đồng ngân sách với screening mới |
| Five-seed B | 256 ảnh | 803 ảnh/3.632 QA lặp trên seed 42–46 | Không nhân số ảnh độc lập lên năm lần |
| LaPA shared/native và PF comparison | Cohort nhỏ đã khóa tương ứng | 1.432 QA, canonical 321 ảnh | Backbone/endpoint khác B generative |
| FAVP adapted | 32 canonical TRAIN | 8 model-validation | TRAIN 31 no/1 yes; validation 6 no/2 yes |
| Lens adapted | 28 TRAIN hợp lệ từ 32 ứng viên | Cùng 8 model-validation đã dùng | TRAIN 27 no/1 yes; loại 4 case không mask, không thay thế |

Việc chọn 32/8 theo ID cố định giúp tránh chọn theo outcome, nhưng không đảm bảo cân bằng hoặc đại diện. Sự lệch lớp của cohort nhỏ là một hạn chế trực tiếp của pilot; không được hậu kiểm tái cân bằng rồi coi đó là cùng thí nghiệm.

## 3.5 Sinh visual prompt và các chế độ Lens

Trong mã v3, `draw_prompt` hỗ trợ contour, box, circle và mask overlay. Contour vẽ biên vùng; box vẽ hộp bao; circle bao vùng theo hình học; mask overlay trộn màu trên vùng. Nếu mask rỗng, ảnh không được biến thành một ROI giả. Có hàm hỗ trợ một kiểu prompt không có nghĩa đã thực hiện quality ablation cho mọi kiểu.

Các chế độ Lens gồm **unicolor** tô cùng màu trên hợp các mask, **multicolor** tô màu khác nhau cho từng mask và **masked** giữ vùng mask, đặt nền ngoài vùng về đen. Trong helper v3, hệ số trộn mặc định lấy từ phân phối quanh 0,5 và được giới hạn; giá trị, seed và recipe thực tế phải đọc từ từng run. Các view làm thay đổi thông tin và phân phối ảnh, nên cần đối chứng original/placebo hoặc cùng-view loss-off.

Trong FAVP nguồn, automatic SAM-Med2D theo ngưỡng nguồn không giữ proposal nào trên 256 TRAIN và 8 validation đã xét. Trong Lens nguồn, SAMHuge tạo view hợp lệ cho 28/32 TRAIN candidates. Không thể dùng coverage của SAMHuge để suy ra SAM-Med2D phải có coverage tương tự, hoặc lấy GT box thay vào nhánh FAVP để báo đã chạy nguyên bản.

# 4 Thiết kế và cài đặt hệ thống

## 4.1 Kiến trúc baseline B

```text
Ảnh gốc → EXIF RGB → letterbox 224 → BiomedCLIP ViT + visual LoRA
                                            │
                           ┌────────────────┴────────────────┐
                           │                                 │
                    Token ảnh 197×768                 QFormerLite
                           │                          32 query×768
Câu hỏi → PubMedBERT → Fusion hai chiều                      │
                           │                                 │
                     Vector fusion 768 ──────────────────────┤
                           │                         Prefix 33×896
                   ┌───────┼────────┐                        │
                Closed   Fracture  Type            Qwen + LoRA → OPEN
                 head      head    head
```

Sơ đồ biểu diễn đường tính toán chính. QFormer và fusion là hai đường xử lý liên quan nhưng không phải tất cả token đều lần lượt đi qua QFormer rồi mới tới fusion. Nhánh closed và fracture dùng vector fusion; Qwen phục vụ sinh câu trả lời mở. [E3]

| Khối | Cài đặt baseline | Đầu vào | Đầu ra |
|---|---|---|---|
| Tiền xử lý | EXIF transpose, RGB, letterbox, normalize | Raster gốc | B×3×224×224 |
| Visual encoder | BiomedCLIP ViT-B/16, trunk frozen + LoRA r4 | Tensor ảnh | B×197×768 |
| Text encoder | PubMedBERT thuộc BiomedCLIP, frozen | Câu hỏi, tối đa 48 token theo config | B×T×768 và mask |
| Fusion | Self-attention rồi cross-attention hai chiều | Token ảnh/văn bản | Vector B×768 |
| QFormerLite | 2 tầng, 8 heads, 32 learned queries | Token thị giác | B×32×768 |
| Prefix bridge | MLP từ 768 sang hidden Qwen 896 | 32 queries + 1 fusion token | B×33×896 |
| LLM | Qwen2.5-0.5B-Instruct, LoRA r8 q/v | Prefix + câu hỏi | Chuỗi OPEN, cap 32 token theo config |
| Closed head | MLP phân lớp | Fusion | N logits, N lấy từ TRAIN vocabulary |
| Fracture head | MLP nhị phân | Fusion | Logit và sigmoid p_yes_raw |
| Answer-type head | MLP nhị phân | Fusion | Tín hiệu routing kết hợp quy tắc |

Không phải mọi dự đoán của hệ thống đều được LLM sinh. Canonical fracture được đọc từ fracture head. Điều này làm endpoint rõ ràng hơn nhưng cũng đòi hỏi kiểm tra routing: trả lời đúng một câu paraphrase không mặc nhiên chứng minh cùng đường suy luận với câu canonical.

## 4.2 Prompt đầu vào và output contract

Baseline B tắt T/V/L/P và không đưa region metadata vào câu hỏi. Tuy nhiên, wrapper LLM vẫn là `Question: {question} Answer:` và soft prefix thị giác vẫn tồn tại. Vì vậy B là “tắt các module prompt khảo sát”, không phải “không có bất kỳ prompt nào”. Wrapper hiện tại không phải toàn bộ Qwen chat template.

Public prediction tối thiểu nhận danh sách ảnh PIL RGB và danh sách câu hỏi. Kết quả có answer, answer_type, confidence; endpoint nhị phân có p_yes và, khi dùng fracture head canonical, p_yes_raw. Confidence không mặc nhiên là xác suất đã hiệu chuẩn cho lâm sàng. Không truyền đáp án, GT answer type hay annotation box vào đường suy luận để giúp model chọn output.

Đầu vào phù hợp nhất với hợp đồng đã kiểm chứng là ảnh X-quang gốc, đúng hướng, chưa bị cắt mất vùng quan trọng và câu hỏi tiếng Anh ngắn thuộc phạm vi dataset. Đây là lựa chọn phù hợp phân phối đã thử, không phải một tối ưu toàn cục đã được chứng minh. Câu tiếng Việt qua lớp chuyển đổi của UI cần đánh giá riêng; không lấy kết quả tiếng Anh làm bằng chứng chất lượng hỏi đáp tiếng Việt tự do.

## 4.3 Fusion và nén token trong nhánh Lens v3

`LensFusion` gắn view embedding, chạy Transformer trên chuỗi token của nhiều view, đưa về cấu trúc view rồi lấy trung bình theo view. Với grid 14 × 14 và hệ số 2, mã dùng **pixel_unshuffle**: gom từng khối 2 × 2 vào chiều kênh, giảm grid xuống 7 × 7; linear projection đưa 4d về d. Đầu ra gồm 49 patch token và 1 CLS, tổng 50 token.

Tên “pixel shuffle” thường được dùng rộng trong thảo luận nén không gian, nhưng thao tác cài đặt ở đây là chiều space-to-depth của `pixel_unshuffle`. Nó không tự phân biệt patch chứa tổn thương, và không bảo đảm giữ mọi thông tin sau projection. Baseline B tắt Lens nên không được gán lợi ích của khối này cho kết quả B.

Pipeline Lens dùng SmolVLM sau cùng có connector và pooling nguồn riêng. Kiểm tra processor thật xác nhận một case có 17 tile mỗi view, 51 tile cho ba view ở cấu hình 2048; pilot gradient/quality dùng long-edge 1024 để phù hợp RAM. Bằng chứng chạy đúng helper cần đến actual forward, không chỉ sự hiện diện của một hàm tên pixel shuffle trong source.

## 4.4 Latent prompt và graph

Port GM/LF lấy bank từ TRAIN và kết hợp latent signal với biểu diễn đa phương thức. Bank đáp án đúng không được truy cập theo target validation để chọn prompt. Random bank, unconditioned hoặc empty bank là các can thiệp khác nhau; empty bank có thể thay đổi capacity và luồng dropout nên không phải đối chứng ngữ nghĩa thuần túy.

PF đưa cấu trúc graph vào prompt. Audit source phát hiện đường GAT của tác giả cho phép nonedge giao tiếp theo cách tính đã cài; masked-neighbor là một biến thể thuật toán cần đặt tên riêng. Residual graph-query giúp truyền tín hiệu nhưng cũng thay mô hình. “Graph off” cần bypass thực sự, không chỉ đặt adjacency bằng zero rồi mặc định không còn tác động.

## 4.5 Quy trình huấn luyện và khóa thực nghiệm

Chuỗi chuẩn bị gồm kiểm tra manifest → dựng fresh common initialization → kiểm tra real forward/backward → kiểm tra gradient và thay đổi tham số → save/reload trong process mới → khóa cohort, config, schedule, dependency → chạy fixed budget → đánh giá LAST → replay → audit metric → bootstrap → xem biểu đồ → closure seal.

Khởi tạo downstream không được lấy checkpoint gate đã update. Gate dùng để chứng minh cơ học optimizer; quality phải bắt đầu từ PRE updates0 được khóa riêng. Shared initialization phải đối chiếu tensor chung và nhãn/vocabulary tương thích. Số bước bằng nhau không đồng nghĩa FLOPs bằng nhau khi một nhánh có thêm view, encoder hoặc visual forward.

ROCO caption alignment đã chạy mới; PMC alignment mới không đạt quality gate nên không được dùng làm initialization downstream cuối. Grouped B dùng lineage đã audit, hai epoch và LAST cố định. Các pilot bốn epoch hay native adaptations có protocol khác và không được gộp thành một lịch huấn luyện duy nhất.

## 4.6 Cấu trúc mã nguồn và artifact

| Vị trí | Vai trò |
|---|---|
| `refactor_v2/src/data/` | Loader, split, prompts và biểu diễn dữ liệu |
| `src/preprocessing.py` | Canonical orientation và resize/letterbox |
| `src/models/encoders.py` | BiomedCLIP và PubMedBERT |
| `src/models/fusion.py`, `lens_fusion.py` | Fusion, LensFusion và QFormerLite |
| `src/models/latent_prompt.py` | GM/LF port và latent components |
| `src/models/prompt_draw.py`, `segment_prompt.py` | Renderer prompt và segmentation wrapper |
| `src/models/bonevqa.py`, `heads.py`, `losses.py` | Mô hình tổng, routing, heads và objectives |
| `src/train.py`, `configs/` | Train loop và recipes |
| `experiments/`, `scripts/`, `tests/` | Runner, gate, audit và regression |
| `references/` | Source tác giả pinned và native assets |
| `reports/`, `checkpoints/` | Giao thức, output, receipts, failures, seals và trọng số |
| `app/` | API, inference engine, lớp ngôn ngữ và giao diện web |

Source pins được giữ riêng: LaPA `677faf90`, FAVP `b43941ba`, Lens `22a94007`. Có 36 native asset đã audit tổng 23.651.792.013 byte. Hash xác nhận nội dung nhận được; không tự xác minh tính tương thích stage training hoặc quyền phân phối lại.

## 4.7 Ứng dụng web hội thoại

Ứng dụng FastAPI có endpoint health, examples, segment, ask, feedback và history. Giao diện nhận ảnh/câu hỏi, hiển thị câu trả lời, confidence, latency, prompt image, Lens image, mask count và detector boxes nếu có. Lớp ngôn ngữ thực hiện chuyển câu hỏi sang tiếng Anh và chuyển đáp án hiển thị khi phù hợp. [E4]

Mã `/api/ask` gọi engine với ảnh và câu hỏi hiện tại rồi lưu lịch sử theo session. Lịch sử hiển thị chưa được truyền toàn bộ như ngữ cảnh đa lượt vào model ở đường gọi này. Do đó mô tả chính xác là giao diện có lịch sử hội thoại cho các lượt hỏi đáp ảnh, chưa có bằng chứng một hệ thống suy luận hội thoại dài hạn.

API còn cho phép boxes và prompt_kind, trong khi evaluated baseline contract chỉ dùng ảnh/câu hỏi. Chế độ tương tác có box phải được tách khỏi benchmark tự động. Mock engine chỉ phục vụ phát triển UI khi bật rõ ràng; startup phải từ chối checkpoint thiếu hoặc không tương thích. Kiểm thử UI không thay thế kiểm định model; báo cáo này không chứng nhận deployment production, bảo mật dữ liệu bệnh nhân hay chất lượng dịch tự do.

# 5 Thực nghiệm và đánh giá

## 5.1 Môi trường và cấu hình

Máy thực nghiệm có 24 GiB RAM; reference quality chủ yếu dùng CPU với tám thread và model runtime offline. Benchmark nhỏ cho thấy CPU8 phù hợp hơn CPU10/12 trong fixture đã thử. MPS có quan sát tăng tốc nhưng không đạt parity cho mọi đường OPEN/resume, nên không được tự động thay CPU reference. Không có benchmark latency/RAM đồng điều kiện đủ để xếp hạng tuyệt đối ba phương pháp.

Runtime v3 dùng Torch 2.2.2 và Transformers 4.40.2 trong môi trường BoneMedVQA cũ. Lens được tách môi trường với Torch 2.6, Transformers 5.0, PEFT 0.18.1 và NumPy 2.2.6; môi trường cũ giữ nguyên. Phiên bản Lens đầu tiên lỗi pyarrow/NumPy được bảo toàn, không ghi đè. Các model asset đã tải trước, không tự tải trong quality inference.

| Thử nghiệm | Budget | Thiết lập chính |
|---|---|---|
| B grouped outer | Seed42, 2 epoch/fold, LAST | T/V/L/P off; CPU reference; calibration tách riêng |
| Common-cohort v3 screen | Seed42, 4 epoch/292 update | 256 TRAIN; batch4/accum4; AdamW 1e-4, wd0,01, clip1 theo protocol |
| Native-backbone LaPA | 5 arm, 4 epoch/292 update mỗi arm | Frozen image/text; base LR5e-6, MM×5, head×50; warmup29 |
| FAVP adapted | 32 TRAIN, 2 epoch/64 update | Batch1; AdamW3e-5; warmup6; greedy20; 4h cap |
| Lens adapted | 2 arm, mỗi arm 42 microbatch/12 update | Batch2/accum4; 1 alignment+2 QA epoch; AdamW2e-4; greedy32; 4h true-wall cap |

Các hyperparameter này không phải cấu hình khuyến nghị chung. Tính chất fixed-budget cho phép diễn giải kết quả mà không chọn thêm epoch sau khi xem validation.

## 5.2 Huấn luyện upstream và lựa chọn initialization

| Run | Kết quả ghi nhận | Diễn giải |
|---|---|---|
| ROCO mới, 2 epoch | Validation loss 2,3638 → 2,2985; 19.556 TRAIN/9.903 validation; driver 23.321 giây | Alignment có lineage và bridge hợp lệ; chưa chứng minh lợi ích downstream nhân quả |
| PMC mới, 2 epoch | Loss 2,7284 → 2,6885; OPEN exact 8,13% → 8,27%, 124/1.500; driver 20.664 giây | Không đạt quality gate; không chọn làm initialization cuối |
| PMC objective control, 128 QA | Cả hai objective 4/128 với ảnh đúng, 1/128 ảnh sai; 127/128 câu trả lời giống nhau giữa điều kiện ảnh | Tín hiệu phụ thuộc ảnh rất yếu trong fixture; bỏ contrastive không giải quyết chất lượng |
| PMC memorization | 16/16 TRAIN sau 100 update | Fit được tập nhỏ; không chứng minh generalization |
| B fresh/ROCO nhỏ, 76 step | QA 69,01%/72,54%; always-no 75,35% nhưng sensitivity0 | Accuracy thô không đủ chọn initialization |
| B seed42–44 nhỏ | Fresh mean AUROC0,8807; ROCO0,8845 | Heads/init khác nhau hạn chế suy luận causal về ROCO |

Timer từng driver không phải tổng thời gian dự án. Candidate config hướng tới upstream không được tính như đã pretrain hoàn chỉnh. Có 23 recipe đã audit cấu trúc trong một giai đoạn, không có nghĩa có 23 quality comparison hoàn tất. [E2]

## 5.3 Kết quả baseline trên grouped outer

| Endpoint | Raw B | B sau hiệu chuẩn |
|---|---:|---:|
| Overall QA exact | 90,00% | 90,17% |
| CLOSED exact | 90,91% | 91,21% |
| OPEN exact | 88,85% | 88,85% |
| Fracture sensitivity | 55,94% | 50,35% |
| Specificity | 94,57% | 96,69% |
| Balanced accuracy | 75,26% | 73,52% |
| AUROC | 0,88399 | 0,88028 |
| AP | 0,71627 | 0,70479 |
| Brier ↓ | 0,09562 | 0,08891 |
| ECE10 ↓ | 0,05728 | 0,02875 |
| TP/FN/TN/FP | 400/315/3.119/179 | 360/355/3.189/109 |

| Outer fold | BA raw | BA calibrated |
|---|---:|---:|
| 0, reuse lịch sử có công bố | 72,66% | 71,50% |
| 1 | 74,27% | 71,49% |
| 2 | 79,24% | 76,27% |
| 3 | 73,66% | 73,25% |
| 4 | 76,46% | 75,10% |

Raw BA có CI 95% 73,16–77,05%; calibrated 71,44–75,38%. Chênh calibrated−raw là −1,74 điểm phần trăm, CI [−2,65; −0,91]. Bootstrap 2.000 lần theo component trong từng fold, seed20261021, giữ model/calibrator cố định. Hiệu chuẩn dùng LogisticRegression cố định trên split calibration; ngưỡng cả hai là 0,5. [E5]

Mô hình raw bỏ sót 315/715 ảnh gãy. Hiệu chuẩn giảm 70 false positive nhưng tăng 40 false negative. Vì vậy, overall QA tăng 0,17 điểm và Brier/ECE tốt hơn không đủ biện minh chọn bản calibrated cho endpoint phát hiện gãy.

## 5.4 Loss và capacity diagnostics

Pilot native/uniform/rarity LAST296 cho QA 113/142, 112/142 và 114/142; chênh rarity−B có CI chứa 0. Khi tái đánh giá toàn 1.432 QA, sequence-uniform có tín hiệu mô tả, nhưng full-TRAIN hai epoch/1.438 update không xác nhận: native QA90,08%, OPEN88,69%; uniform QA89,94%, OPEN88,06%; sensitivity đều64,91%. Các paired interval được báo cáo đều chứa0.

Trên TRAIN 2.568 ảnh, sensitivity native68,56% và uniform67,25%. Điều này cho thấy vấn đề không chỉ xuất hiện ở unseen validation. Tuy nhiên, gradient probes chưa chứng minh LM loss xung đột là nguyên nhân; không có cơ sở loại loss hoặc áp dụng gradient surgery chỉ từ giả thuyết đó.

Frozen-feature fracture-head pilot 10 epoch/410 update nâng sensitivity model-validation từ64,91% lên84,21%, giảm specificity93,18% xuống83,71%; BA79,05% lên83,96% nhưng CI chênh [−0,33;10,39] điểm chứa0. Có thêm11 TP và25 FP, calibration xấu hơn. Đây là trade-off thăm dò, không phải candidate được promote. [E2]

## 5.5 Screening T V L P và năm seed

| Arm cùng cohort 256 TRAIN | AP LAST4 |
|---|---:|
| B | 0,668688 |
| V-real | 0,669935 |
| V-location-placebo | 0,674111 |
| P-TRAIN | 0,668107 |
| P-random | 0,683351 |
| P-unconditioned | 0,662946 |
| L-real | 0,665445 |
| L-original | 0,676976 |
| L-placebo | 0,657304 |

Quy tắc screening yêu cầu vượt B và từng own-control ít nhất0,005 AP. Không module nào đạt. V-real−placebo = −0,004176, CI [−0,016126;0,003750]. L có146 alignment+146 QA update, B có292 QA update; matched comparison có ý nghĩa nhất giữa các L control cùng phase. [E6]

Textual prompt semantic, irrelevant token-matched và paraphrase ở các screen trước cho AUROC0,88424/0,88417/0,88404; paired AP/AUROC interval chứa0. Chưa chứng minh lợi ích ngữ nghĩa của template. Đây không phải thử nghiệm đầy đủ mọi cách đặt câu hỏi.

Five-seed B trên cohort nhỏ, seed42–46, cho AP0,51202 ±0,01694; raw sensitivity46,43%, specificity88,67%, BA67,55%. Sau hiệu chuẩn, sensitivity26,85%, specificity95,94%, BA61,40%. Cùng803 ảnh được đánh giá lặp trên năm model; không được coi đó là4.015 ảnh độc lập. Số AP này không phải sự suy giảm từ grouped AP0,71627 vì lượng TRAIN và thiết kế khác nhau. [E7]

## 5.6 Chất lượng định vị và đóng góp của visual prompt

Trên321 ảnh model-validation, có57 ảnh gãy;49 có proposal,40 có best-box IoU≥0,25 và26 đạt≥0,5. Với tám false negative của B, bảy có proposal và ba đạtIoU≥0,5. V-real cứu0/8 và mất thêm hai true positive; placebo cũng cứu0/8. Như vậy thiếu proposal không giải thích hết các ca bỏ sót. Đây là image-level best-box hit, không phải lesion-level recall. [E8]

Kiểm tra tọa độ phát hiện hai ảnh EXIF khác orientation; cả hai không có GT box. Sau đối chiếu canonical dimensions và bounds, cohort được phân tích. Việc sửa coordinate gate là cần thiết trước khi tính overlap, nhưng không tự nâng quality của detector.

Pilot local-token tiếp theo có real_aux AP0,694109, placebo_aux0,708043 và real_noaux0,732725. Real_aux−placebo CI[−0,077238;0,044088]; real_aux−real_noaux CI[−0,078240;−0,000557]. Tín hiệu gợi ý cần nghiên cứu tương tác auxiliary loss, nhưng real_noaux thiếu placebo/grid cùng noaux nên chưa chứng minh lợi ích vị trí thật. [E9]

Pilot spatial head có AP QA real0,679533, placebo0,674547, uniform0,677791 và off0,668738; real−placebo0,004986 chưa đạt margin0,005, không được làm tròn thành đạt. Trong khi đó box-patch AP real0,384351 cao hơn placebo0,063555 và off0,183074. Head đã học thông tin vị trí liên quan box nhưng không nằm trực tiếp trên public answer path; lợi ích định vị phụ chưa chuyển thành lợi ích QA đủ tiêu chí. [E10]

## 5.7 Consistency bank và shared-backbone LaPA

| Bank trong port P v3 | Consistency on AP | Consistency off AP |
|---|---:|---:|
| TRAIN | 0,668107 | 0,673248 |
| Random | 0,683351 | 0,660443 |
| Empty | 0,662089 | 0,660904 |

TRAIN−random ở consistency-off là+0,012805, CI[−0,005122;0,033738]; ở on là−0,015244, CI[−0,036011;0,002006]. Không có bằng chứng bank TRAIN mang lợi ích ngữ nghĩa. Hiệu ứng consistency thay đổi theo bank; không thể nói consistency luôn có hại. [E11]

Shared-backbone source-oriented LaPA có carrier0,406441; TRAIN-off0,303897; TRAIN-on0,448130; random-off0,650512; random-on0,491071. TRAIN−random-off CI[−0,462567;−0,209211], on CI[−0,165642;0,062736]. Random-on có sensitivity0 tại0,5. Pilot này dùng classifier41 class và khởi tạo cross-modal khác; không so AP/accuracy của nó với B như cùng thí nghiệm. [E12]

## 5.8 PF tái dựng và LaPA dùng backbone gốc

| PF shared-backbone | AP |
|---|---:|
| Author-route GAT | 0,373552 |
| Masked-neighbor | 0,509257 |
| Residual real graph | 0,417577 |
| Residual rewired graph | 0,318279 |

Real−rewired AP+0,099297 có CI[−0,004839;0,222592], chưa xác lập lợi ích ở endpoint chính. Secondary closed-answer accuracy tăng0,050978 với CI dương, nhưng không thay được primary fracture AP. Masking và normalization thay cùng lúc nên không cô lập riêng hai yếu tố. [E13]

| LaPA native-backbone adaptation | AP LAST |
|---|---:|
| Carrier | 0,660138 |
| GM+LF | 0,526043 |
| Author PF | 0,430638 |
| Residual PF real | 0,430453 |
| Residual PF rewired | 0,603072 |

Năm nhánh đều4 epoch/292 update, replay1.432 record chính xác. AuthorPF−GMLF = −0,095405, CI[−0,172551;−0,011226]; residual−author gần0; residual real−rewired = −0,172620, CI[−0,257892;−0,085320]. Không promote PF. M3AE thật đã được nạp, nhưng frozen encoder, graph409 và endpoint41class khiến đây vẫn là adapted pilot. Hai study PF cho kết quả khác nhau cũng minh họa phụ thuộc backbone/initialization; không được lựa chọn study thuận lợi rồi tuyên bố graph có ích chung. [E14]

## 5.9 FAVP dùng tài nguyên gốc

Native SAM/EVA/QFormer/Vicuna đã qua forward, gradient/update, frozen-state guard và fresh replay. Tối ưu CPU chuyển phép frozen linear từ tính FP16 rất chậm sang FP32 tạm thời đã được đo và công bố sai khác số học. Nó không phải bitwise-equivalent với phép tính FP16 ban đầu.

Pilot32 TRAIN/8 validation,64 update đạt PRE0/8 → epoch1 vàLAST6/8. Tất cả answer cuối đều mang đáp án “no”, bằng constant TRAIN-majority. Tám output còn chạm giới hạn20 token. Không có bằng chứng lợi ích điều kiện theo ảnh hoặc local branch; toàn bộ264 ảnh đã xét không có proposal SAM-Med2D vượt threshold nguồn, nên local-on/off có đầu vào tương đương. [E15]

Do thiếu projection stage1/2 gốc, không thể diễn giải đây là sự thất bại của FAVP công bố. Mặt tích cực được xác minh là full trainable route và replay hoạt động; mặt chất lượng là pilot nhỏ đã collapse về prior đa số. Tổng thời gian thực thi được báo cáo khoảng185,5 phút, không phải ETA chung cho FAVP ở mọi máy.

## 5.10 Lens dùng tài nguyên gốc

| Giai đoạn | CE-only strict EM | Lens-loss strict EM |
|---|---:|---:|
| PRE | 0/8 | 0/8 |
| Sau alignment | 0/8 | 0/8 |
| QA epoch1 | 5/8 | 5/8 |
| LAST QA epoch2 | 6/8 | 6/8 |

Cả hai arm dùng28 TRAIN hợp lệ, cùng ba view, cùng fresh initialization; mỗi arm12 update. Tất cả492 LoRA tensor thay đổi, frozen state giữ nguyên và fresh replay tám record khớp. LAST đều trả lời “No”: sensitivity0, specificity1, BA0,5, bằng constant baseline. Lens−CE exact difference0, CI[0,0] do tám paired outcomes giống nhau; không phải chứng minh equivalence tổng quát. [E16]

PRE có secondary leading-binary5/8 dù strict EM0 vì trả lời dài. Secondary BA từ0,75 ở PRE xuống0,5 tạiLAST. Một phần gain strict EM phản ánh học định dạng ngắn, không phải tăng năng lực nhận biết gãy. Không đổi parser hậu kiểm để chọn kết luận thuận lợi.

Resource adaptation gồm processor1024 thay2048, SAMpointbatch8 thay64 và checkpointingCPU. Region alignment dựa nhãn CSV hiện có, không fallback cơ quan giả. Lens-loss thực hiện thêm visual forward, khác compute/RNG với CE-only; vì vậy đây không phải thí nghiệm dropout bitwise-matched. Bốn case không đủ mask bị loại không thay thế.

Worker CE dùng2.039,35 giây, Lens3.654,12 giây; peak RAM tương ứng10,30/11,35 GiB. Coordinator ghi95,92 phút compute timer, trong khi elapsed thật161,04 phút do máy sleep. Guard ngoài dependency đã khóa giữ deadline4 giờ theo wall clock và xác nhận hoàn tất trong hạn. Thời gian ngủ được công bố thay vì kéo dài budget hoặc khởi động lại để che gián đoạn.

## 5.11 Ma trận ablation đã thực hiện và phần còn thiếu

| Câu hỏi | Đối chứng đã chạy | Trạng thái kết luận |
|---|---|---|
| T có ngữ nghĩa hữu ích? | Raw/semantic/irrelevant matched/paraphrase ở screen | Chưa xác lập lợi ích; không phải mọi prompt |
| V dùng vị trí đúng? | Real/location-placebo; B reference | Chưa vượt matched placebo |
| Local token và auxiliary loss? | Real_aux/placebo_aux/grid_aux/real_noaux | Có tín hiệu tương tác; thiếu noaux-placebo để kết luận semantic |
| L alignment trong v3? | Real/original-teacher/placebo cùng phase | Chưa vượt controls đủ rule |
| Spatial supervision? | Real/placebo/uniform/off | Localization head tốt hơn, QA chưa đạt rule |
| P bank/consistency? | TRAIN/random/empty × on/off | Đã bổ sung sáu cell; chưa thấy TRAIN semantic benefit |
| PF graph semantics? | Real/degree-preserving rewired ở residual route | Shared CI primary chứa0; native adaptation real kém rewired |
| Native Lens contrastive losses? | CE-only/loss-on, cùng view | Cùng6/8, chưa có gain |
| Lens từng mode unicolor/multicolor/masked? | Có code/view và technical checks | Chưa có quality factorial riêng cho mọi mode |
| Pixel compression cần thiết? | Có implementation và real processor check | Chưa có matched quality compression-on/off |
| FAVP local có ích? | Native coverage census và pilot fallback | Không có local proposal để tạo informative on/off comparison |
| Full V+L+P hay LaPA+FAVP+Lens? | Chưa có integrated matched factorial | Chưa có kết luận synergy; không có “bản ghép thắng” |
| Full same-cohort image/question/geometry-only? | Có diagnostic cục bộ và constant baseline | Chưa có bộ causal control đầy đủ ở mọi cohort cuối |
| External patient-independent confirmation? | Chưa có | Không tuyên bố clinical validation |

“Ablation đầy đủ” trong báo cáo có nghĩa công khai đầy đủ những gì đã làm và còn thiếu. Không thể gọi ma trận thực nghiệm hiện tại là exhaustive factorial. Đặc biệt, huấn luyện loss-off khác với tắt một module ở inference; Lens training loss không có cùng ý nghĩa như một nút on/off trong mô hình deployed.

## 5.12 Mô hình có thật sự dùng ảnh không

Có ba cấp bằng chứng. Thứ nhất, gradient/update và image-swap fixture xác nhận ảnh có đường tới logits. Thứ hai, cần chứng minh ảnh đúng tốt hơn ảnh sai hoặc prior chỉ-câu-hỏi trên cùng cohort. Thứ ba, cần chứng minh tín hiệu đến từ vùng tổn thương đúng chứ không từ view, thiết bị, kích thước hay dấu hiệu nền.

Baseline v3 có bằng chứng cấp một và một số intervention cục bộ. Tuy nhiên, chưa có bộ kiểm chứng cấp hai/ba hoàn chỉnh cho mọi final cohort. FAVP/Lens pilot thì output all-no đủ để nói chưa quan sát lợi ích sử dụng ảnh ở endpoint tám ca đó; không đủ để nói mọi internal visual representation bằng zero. Spatial-head AP tốt và answer-head không cải thiện cho thấy “học được vị trí” và “dùng vị trí để trả lời” là hai yêu cầu khác nhau.

Một thiết kế tiếp theo cần khóa trước image-swap có kiểm soát vùng/view, blank-image, question-only, geometry-only và occlusion vùng thật/đối chứng, đồng thời báo both logit changes và outcome differences. Các phép này là hướng nghiên cứu, không phải kết quả đã chạy thêm trong lần xuất báo cáo.

## 5.13 Phân tích lỗi trong quá trình phát triển

| Nhóm lỗi | Phát hiện và xử lý có hồ sơ | Bài học |
|---|---|---|
| Provenance upstream | PMC train_lon overlap, checkpoint thiếu manifest | Không dùng checkpoint không rõ lineage làm bằng chứng độc lập |
| Loader/schema | Fallback split, malformed row, đường dẫn dữ liệu cũ | Fail-closed; giữ input/role/hash rõ ràng |
| Routing/loss | Fracture endpoint, token/sequence reduction, bridge gradient | Đối chiếu loss thủ công và actual gradients |
| Runtime | MPS parity, checkpointing/resume, OpenMP | Chọn reference đã kiểm tra; không đánh đổi đúng đắn lấy tốc độ im lặng |
| Coordinate | EXIF và bounds trước overlap | Hệ tọa độ phải thống nhất trước metric định vị |
| LaPA/PF | Alias checkpoint, text-wrapper path, GAT nonedge, signal attenuation | Giữ author route và variant riêng; không ghi variant như nguyên bản |
| FAVP | EVA F.linear dùng qkv.weight bỏ qua LoRA wrapper | Sửa route thực thi bias-aware, kiểm tra gradient mọi nhóm |
| Vicuna | Legacy RoPE, API mmap/cache_position | Compatibility phải được xác nhận bằng native forward/replay |
| Lens runtime | PyArrow/NumPy không tương thích | Môi trường mới riêng, giữ failed version |
| Lens receipt | JSON không serialize được set sau render | Phục hồi receipt từ artifact đã có, không giả thời gian/raw count đã mất |
| Wall budget | Máy đóng nắp làm timer không phản ánh sleep | Guard wall-clock ngoài source khóa, giữ nguyên deadline |

Các failure artifact không bị xóa. Regression suite tại technical seal có104/104 test đạt; các mốc test trước đó không được cộng dồn thành số test độc lập. Một test đạt chỉ chứng minh điều kiện được assert; không dùng tổng số test làm thay thế hiệu quả khoa học.

## 5.14 So sánh với nghiên cứu liên quan

| Hướng | Mục tiêu gốc | Điều đã thực hiện trong BoneVQA | Khoảng cách còn lại |
|---|---|---|---|
| LaPA | Latent/knowledge-assisted Med-VQA | GM/LF controls, graph variants, native M3AE pilot | Graph/node-order gốc, full fine-tune, original endpoint/budget |
| FAVP | Adaptive local visual prompt và generative QA | Native full route, LoRA, generation/replay, pilot | Stage1/2 projection provenance, informative local coverage, representative cohort |
| Lens | Localization-aware views/alignment/compression | Native SAM/Smol, true contrastive batch, loss-on/off | Original-scale model/resolution/supervision và view/compression factorial |
| BoneVQA v3 | QA X-quang xương với direct fracture endpoint | Grouped evaluation, prompt controls, audits/calibration | External patient-independent cohort và causal image-grounding study |

Các paper báo kết quả trên dataset và protocol của họ. Báo cáo không lấy accuracy VQA-RAD/SLAKE/DME từ paper ghép với FracAtlas thành leaderboard. Sự khác biệt gồm backbone, pretraining, supervision, cohort, số tham số trainable, compute và cách chuẩn hóa answer. Không có cơ sở kết luận v3 vượt paper gốc, hoặc paper gốc thất bại từ các adapted pilot âm.

## 5.15 Phạm vi thống kê và giới hạn kết luận

Các bootstrap đa số dùng2.000 paired component draws và seed phân tích được khóa riêng. Chúng mô tả biến thiên lấy mẫu của cohort đã có, giữ model cố định. Nhiều contrast là exploratory, không điều chỉnh multiple comparisons; các điểm thuận lợi đơn lẻ không được nâng thành xác nhận giả thuyết chính.

Model-validation đã được dùng nhiều vòng development. Eight-case pilot có khả năng phát hiện hiệu ứng rất hạn chế và class imbalance mạnh. Các câu template có cấu trúc lặp, nên OPEN exact cao không tương đương khả năng viết báo cáo X-quang tự do. Unknown patient relation, unknown backbone pretraining overlap, thiếu assets gốc và resource adaptation đều giới hạn khả năng khái quát.

Không có báo cáo test độc lập hợp lệ cho mọi nguồn chỉ vì dataset có thư mục test. PMC filtered test là manifest đã chuẩn bị; VQA-RAD lịch sử có ảnh chung; FracAtlas outer là development-informed; Lens/FAVP tám case là model-validation đã tái dùng. Đây là các trạng thái bằng chứng khác nhau và phải giữ tên đúng.

# 6 Kết luận và hướng phát triển

## 6.1 Kết quả đạt được

BoneVQA v3 đã hình thành một pipeline có thể train và tái lập với image/text encoders, multimodal fusion, visual prefix và LLM, đồng thời tách endpoint gãy khỏi điểm QA tổng hợp. Hệ thống có kiểm soát split theo known component, TRAIN-only vocabulary, kiểm tra gradient/update, fresh-process replay và artifact seals. Các thử nghiệm nguồn LaPA, FAVP, Lens đã được chạy thật trong phạm vi công bố, gồm cả backward và checkpoint reload, thay vì chỉ có sơ đồ hoặc wrapper.

Giá trị khoa học rõ nhất nằm ở kết quả đối chứng: prompt có kết nối kỹ thuật nhưng chưa chứng minh semantic utility; định vị phụ có thể cải thiện mà QA không cải thiện; calibration có thể tốt hơn về xác suất nhưng xấu hơn ở decision threshold; và75% accuracy có thể chỉ là prior lớp đa số. Những kết quả này giúp loại bỏ các diễn giải quá mức và định hướng thí nghiệm tiếp theo.

## 6.2 Hạn chế hiện tại

Baseline còn bỏ sót44,06% ảnh gãy trong grouped outer. Các pilot mới không chứng minh thay thế được baseline. Chưa có full factorial tích hợp, external cohort chưa từng dùng, xác minh patient identity hoặc matched causal grounding benchmark. Bản web là research interface; chưa có clinical/production validation. Missing original assets và pretraining handoff vẫn cản trở exact reproduction.

Vì vậy, lựa chọn hợp lý hiện tại là giữ B làm reference nghiên cứu có phạm vi bằng chứng rộng nhất trong dự án, đồng thời giữ các module như nhánh thí nghiệm. Điều này không khẳng định B tối ưu tuyệt đối về accuracy, tốc độ, chi phí hay kiến trúc.

## 6.3 Hướng phát triển ưu tiên

Ưu tiên đầu tiên là một cohort đánh giá mới có patient grouping và miền ảnh tương thích, được khóa trước mọi lựa chọn model. Nếu chưa có dữ liệu như vậy, mọi phát triển tiếp tục trên cohort hiện hữu phải được gọi rõ là development và không quảng bá như independent confirmation.

Ưu tiên thứ hai là thiết kế matched controls chứng minh sử dụng ảnh có ích. Cần giữ cùng initialization, sampler, head, endpoint và budget; công bố view/token/FLOPs khác nhau. Có thể bắt đầu bằng B, image-shuffle, question-only và constant baseline trước khi mở rộng factorial lớn. Cần đo sensitivity cùng specificity và calibration, thay vì tối ưu một mình exact accuracy.

Ưu tiên thứ ba là nghiên cứu chi tiết ảnh: native resolution/crop tự động, local token và cách đưa spatial signal vào answer path. Các biến thể cần matched placebo và kiểm tra tổn thất thông tin sau compression. Không chọn crop bằng annotation khi public inference không có annotation.

Ưu tiên thứ tư là hoàn thiện provenance phương pháp: lấy đúng graph/node ordering LaPA và checkpoint handoff FAVP nếu có nguồn hợp lệ; xác định rõ cấu hình Lens và supervision. Nếu dùng adaptation, đặt tên và ghi khác biệt ngay từ protocol, không đợi đến khi kết quả âm mới bổ sung giới hạn.

Cuối cùng, nếu hướng tới ứng dụng, cần tách benchmark model khỏi đánh giá UI, bổ sung hỗ trợ ngôn ngữ được kiểm tra, khả năng từ chối trường hợp ngoài phạm vi, quản trị dữ liệu và nghiên cứu người dùng. Các bước đó cần bằng chứng riêng; một demo chạy được không thay thế quy trình đánh giá sử dụng y khoa.

## Tuyên bố tái lập và tính toàn vẹn

Báo cáo này không chạy lại training, không sửa source đã khóa, không chọn thêm seed/epoch/threshold và không tạo nhãn mới. Ba pilot cuối đóng với LaPA52 artifacts/197 dependencies, FAVP25/606 và Lens44/222. Hồ sơ âm và failure vẫn được bảo toàn. Chỉ mục đi kèm bao phủ các báo cáo Markdown nguồn tại thời điểm xuất; nó là danh mục bằng chứng, không phải khẳng định mọi tài liệu lịch sử có cùng độ tin cậy.

# Tài liệu tham khảo học thuật

[R1] Gu T, Yang K, Liu D, Cai W. LaPA: Latent Prompt Assist Model For Medical Visual Question Answering. CVPR Workshops, 2024. [Bài báo](https://arxiv.org/abs/2404.13039), [mã tác giả](https://github.com/GaryGuTC/LaPA_model).

[R2] Yu T, Tong Z, Yu J, Zhang K. Fine-grained Adaptive Visual Prompt for Generative Medical Visual Question Answering. AAAI 39(9), 9662–9670, 2025. [Bài báo và DOI](https://ojs.aaai.org/index.php/AAAI/article/view/33047), [mã tác giả](https://github.com/OpenMICG/FAVP).

[R3] Farooq H, Taj M, Nasim M, Mahmood A. Localization Lens for Improving Medical Vision–Language Models. MICCAI, 2025. [Bản hội nghị](https://papers.miccai.org/miccai-2025/0506-Paper2723.html), [DOI](https://doi.org/10.1007/978-3-032-05114-1_33).

[R4] Dosovitskiy A et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR, 2021. [Bài báo](https://arxiv.org/abs/2010.11929).

[R5] Zhang S et al. BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. [Bài báo và lịch sử phiên bản](https://arxiv.org/abs/2303.00915).

[R6] Kirillov A et al. Segment Anything. 2023. [Bài báo](https://arxiv.org/abs/2304.02643).

[R7] Cheng J et al. SAM-Med2D. 2023. [Bài báo](https://arxiv.org/abs/2308.16184).

[R8] Qwen Team. Qwen2.5-0.5B-Instruct model card. [Nguồn nhà phát hành](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

[R9] Hu EJ et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022. [Bài báo](https://arxiv.org/abs/2106.09685).

[R10] Li XL, Liang P. Prefix-Tuning: Optimizing Continuous Prompts for Generation. ACL-IJCNLP, 2021. [ACL Anthology](https://aclanthology.org/2021.acl-long.353/).

Các mô tả lý thuyết được diễn giải; không sao chép toàn văn công trình. Các con số thực nghiệm BoneVQA lấy từ hồ sơ nội bộ dưới đây, không lấy từ leaderboard paper để thay thế phép đo dự án.

# Phụ lục A Hồ sơ bằng chứng chính

Các đường dẫn tương đối sau tính từ thư mục chứa báo cáo. Bản HTML có liên kết đến cùng các nguồn trong workspace.

- [E1 — Giao thức dữ liệu](DATA_PROTOCOL_STEP1_V3.md).
- [E2 — Tổng hợp audit và thực nghiệm bước 1–3](BAO_CAO_TONG_HOP_BUOC_1_2_3_V3_20261022.md).
- [E3 — Mô hình tổng](../src/models/bonevqa.py), [encoders](../src/models/encoders.py), [fusion](../src/models/fusion.py), [LensFusion](../src/models/lens_fusion.py), [losses](../src/models/losses.py), [config B pilot](step4_localization_v3_1/B.json).
- [E4 — API](../app/api.py), [inference engine](../app/inference.py), [giao diện](../app/static/index.html), [ngôn ngữ](../app/ngon_ngu.py).
- [E5 — Grouped outer đầy đủ](GROUPED_OUTER_RESULTS_V3_1.md).
- [E6 — Screening cùng cohort](step56_screening_v3_1/SCREENING_OUTCOME.md), [V primary contrast](step4_localization_v3_1/STEP4_EXIT_EVIDENCE.md).
- [E7 — Five-seed](step6_confirmation_v3_1/FIVE_SEED_RESULTS.md).
- [E8 — Localization](step7_synthesis_v3_1/LOCALIZATION_FINDINGS.md), [tổng hợp và bootstrap](step7_synthesis_v3_1/RESEARCH_SYNTHESIS.md), [toàn bộ 25 question strata của 19 điều kiện](step7_synthesis_v3_1/QUESTION_RESULTS.md).
- [E9 — Local-token pilot](vnext_local_tokens_v1/V_INTERPRETATION.md).
- [E10 — Spatial-head pilot](lnext_spatial_runs_v1/SUPPLEMENTARY_INTERPRETATION.md).
- [E11 — Bank × consistency](pnext_consistency_bank_runs_v1/SUPPLEMENTARY_INTERPRETATION.md).
- [E12 — Shared LaPA bảng đầy đủ](lapa_shared_quality_v1/REFERENCE_PILOT_RESULTS.md), [diễn giải cuối](lapa_shared_quality_v1/FINAL_INTERPRETATION.md).
- [E13 — PF comparison](pf_comparison_quality_v1/FINAL_INTERPRETATION.md).
- [E14 — Native-backbone LaPA](lapa_native_quality_v1/FINAL_INTERPRETATION.md), [bảng kết quả](lapa_native_quality_v1/REFERENCE_PILOT_RESULTS.md), [closure](lapa_native_quality_v1/closure_seal.json).
- [E15 — FAVP](favp_bounded_quality_v1/FINAL_INTERPRETATION.md), [kết quả](favp_bounded_quality_v1/PILOT_RESULTS.md), [closure](favp_bounded_quality_v1/closure_seal.json).
- [E16 — Lens](lens_bounded_quality_v1/FINAL_INTERPRETATION.md), [kết quả](lens_bounded_quality_v1/PILOT_RESULTS.md), [closure](lens_bounded_quality_v1/closure_seal.json).
- [Tổng kết ba phương pháp](THREE_METHOD_BOUNDED_PILOT_SUMMARY_20261027.md), [phạm vi so sánh](THREE_METHOD_COMPARISON_SCOPE_V1.md), [wall-clock review](LENS_SLEEP_WALL_BUDGET_REVIEW_V1.md).
- [Chỉ mục toàn bộ báo cáo nguồn và hash tại thời điểm xuất](BONEVQA_V3_SOURCE_INDEX_20261027.md).

# Phụ lục B Cách đọc kết quả khi viết bài báo

Trong abstract nên nêu đồng thời QA accuracy và sensitivity/BA, không chỉ90%. Trong phần Methods cần công bố development-informed split, historical fold0 reuse, cách hình thành component và fixed LAST. Trong Results cần tách screening khỏi grouped evaluation và source adaptation khỏi reproduction. Trong Discussion cần giữ các negative controls, majority collapse và giới hạn8case.

Không dùng các cụm “state of the art”, “clinical ready”, “full ablation completed”, “exact reproduction” hoặc “mô hình thực sự hiểu ảnh” khi bằng chứng chưa hỗ trợ. Có thể mô tả công trình là một nghiên cứu hệ thống về khả năng tái lập và đóng góp của prompt trong BoneVQA, với phát hiện rằng các cải tiến kỹ thuật không tự chuyển thành lợi ích phát hiện gãy trong các cấu hình đã kiểm tra. Đây là cách trình bày phù hợp với kết quả thực tế và cho phép nghiên cứu tiếp theo bắt đầu từ một nền bằng chứng rõ ràng.
