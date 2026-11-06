# Phương án tải tài nguyên gốc — chờ duyệt

Ngày 26/10/2026. Chỉ đọc mã, danh mục tệp và HTTP HEAD; chưa tải trọng số, cài môi trường, thuê GPU hoặc khởi chạy huấn luyện. Thứ tự: LaPA/PF → FAVP → Localization Lens.

## Máy hiện tại và quy ước

Đã đọc tại máy: RAM 25,769,803,776 byte (24 GiB), 12 lõi vật lý/12 logic; đĩa còn khoảng 40 GiB tại thời điểm kiểm tra. RAM này không được coi là VRAM NVIDIA. Các mã CUDA gốc không tự nhiên tương đương khi chuyển sang CPU/MPS.

Dung lượng tệp dưới đây dùng GB thập phân. Ngân sách đĩa/GPU là ước lượng lập kế hoạch, không phải cấu hình tối thiểu đã đo hay cấu hình tác giả xác nhận. Dung lượng tải không bao gồm môi trường, dữ liệu, checkpoint đầu ra, optimizer và bản tạm.

## 1. LaPA/PF — ưu tiên đầu tiên

Nguồn chuẩn: [LaPA tác giả](https://github.com/GaryGuTC/LaPA_model), [M3AE tác giả](https://github.com/zhjohnchan/M3AE).

| Tài nguyên | Nguồn được README LaPA chỉ định | Dung lượng xác minh |
|---|---|---|
| Graph 577 nút, adjacency, node tokens/order, answer-bank/lookup assets | [External Data](https://unisydneyedu-my.sharepoint.com/:f:/g/personal/tigu8498_uni_sydney_edu_au/Ev_GsFFY9BpKi-cOHSFIHecBRxYEB-9jjelVen4WUyyEGg) | Chưa xác định; trước đây nhận trang lỗi, chưa nhận asset hợp lệ |
| M3AE pretrained, không phải checkpoint downstream | [Drive do tác giả dẫn](https://drive.google.com/drive/folders/1b3_kiSHH8khOQaa7pPiX_ZQnUIBxeWWn) | Chưa xác minh danh mục/kích thước |
| RoBERTa-base + tokenizer khớp | [Drive do tác giả dẫn](https://drive.google.com/drive/folders/1ouRx5ZAi98LuS6QyT3hHim9Uh7R1YY1H) | Chưa xác minh danh mục/kích thước |

Không đưa ra tổng tải giả chính xác khi chưa có danh mục. Trước tiên giải quyết quyền truy cập/danh mục; nếu cần liên hệ tác giả, soạn thư để người dùng duyệt, không tự gửi. Graph SLAKE 409 nút hiện có không thay cho graph 577 nút. Kiểm tra riêng provenance ngân hàng prompt và từ điển target, không suy ra chúng giống nhau.

Đề xuất ngân sách làm việc tạm: 20–30 GiB đĩa trống dành riêng, RAM máy chạy 32 GiB trở lên; một GPU NVIDIA 24 GiB là mục tiêu thử gate ban đầu, 48 GiB là phương án dự phòng nếu batch/resolution gốc vượt bộ nhớ. Chưa bảo đảm batch gốc vừa 24 GiB. Chỉ benchmark sau khi khóa cấu hình CLIP/RoBERTa/M3AE và giải quyết khác biệt paper/code; không tự giảm batch rồi gọi là tái lập y nguyên.

**Điểm duyệt 1:** chỉ tải khi đã có danh sách tệp, dung lượng, nguồn và provenance; báo lại tổng thực tế trước khi nhận trọng số. Không tự thay bằng mirror hoặc backbone khác.

## 2. FAVP

Nguồn tuyến mô hình: [FAVP tác giả](https://github.com/OpenMICG/FAVP).

| Thành phần | Nguồn | Dung lượng trọng số |
|---|---|---:|
| Vicuna V0 7B | [Vision-CAIR/vicuna-7b](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) | 13,476,966,481 byte, khoảng 13.477 GB; hai shard |
| EVA ViT-G | [LAVIS URL được mã FAVP dùng](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth) | 2,025,249,237 byte, khoảng 2.025 GB; HTTP HEAD |
| Q-Former checkpoint | [LAVIS URL được mã FAVP dùng](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth) | 433,481,467 byte, khoảng 0.433 GB; HTTP HEAD |
| SAM-Med2D vit_b + adapter | [SAM-Med2D chính thức, checkpoint trong bảng Results](https://github.com/OpenGVLab/SAM-Med2D) | Chưa xác minh kích thước tệp được liên kết |
| Checkpoint chuyển tiếp stage 1/2, tokenizer/config cần thiết | Tác giả FAVP và chuỗi tiền huấn luyện tương ứng | Chưa xác minh; không bỏ qua |

Phần trọng số đã biết cộng khoảng **15.936 GB**, chưa gồm SAM-Med2D và checkpoint chuyển tiếp. Checkpoint RAD được README giới thiệu là đã fine-tune RAD; không dùng như pretrained sạch mà chưa kiểm tra exposure. Cấu hình RAD còn trống đường dẫn và epoch; launcher stage 2 có tên config khác README. Phải giải quyết trước khi chạy.

Ngân sách đề xuất: 60–100 GiB đĩa làm việc, RAM 64 GiB; GPU NVIDIA 48 GiB là mục tiêu khởi đầu thận trọng cho gate LoRA/batch1, vẫn phải đo peak memory theo số proposal. GPU 24 GiB có thể đòi hỏi offload/quantization; không mặc nhiên coi đó là bản gốc. Đây không phải ngân sách để tiền huấn luyện lại toàn bộ ROCO/PMC. Nếu không có checkpoint chuyển tiếp hợp lệ, phải duyệt riêng chi phí tái tiền huấn luyện; không tự thực hiện.

**Điểm duyệt 2:** chưa nên tải cả FAVP vào ổ hiện chỉ còn 40 GiB. Chọn ổ ngoài hoặc máy GPU có đĩa phù hợp trước; không xóa kết quả cũ để lấy chỗ.

## 3. Localization Lens

Đây là tài nguyên cho tuyến `src` ở revision đã ghim của [Localization Lens](https://github.com/CVLABLUMS/localizationlens); không khẳng định tuyến này bao phủ mọi thí nghiệm trong paper.

| Thành phần | Nguồn | Dung lượng |
|---|---|---:|
| SmolVLM-256M-Instruct | [HuggingFaceTB chính thức](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/tree/main) | `model.safetensors` 513,028,808 byte; kèm tokenizer/processor/config khoảng 0.52 GB |
| SAM ViT-Huge theo adapter Transformers của source | [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge/tree/main) | `model.safetensors` 2,564,432,288 byte; kèm config khoảng 2.565 GB |

Gói chọn lọc khoảng **3.1 GB**. Không tải thêm ONNX hoặc bản `.bin` trùng trọng số. Không thay SmolVLM bằng SmolVLM2/500M hoặc SAM-Huge bằng MedSAM-base.

Ngân sách đề xuất: 15–25 GiB đĩa, RAM 24–32 GiB; GPU NVIDIA 16–24 GiB để thử sinh mask rồi chạy LoRA ba views theo hai bước tuần tự. Đây là ước lượng, không dùng thông số inference một ảnh để bảo đảm bộ nhớ training ba views. CPU có thể dùng kiểm tra nhỏ nhưng chưa có ETA thực nghiệm. Môi trường hiện tại không được nâng cấp trực tiếp; dựng môi trường riêng sau duyệt. Kiểm tra tile counts/order, batch contrastive thật, supervision hợp lệ và split binding trước release.

## Quy trình chung sau khi được duyệt tải

1. Chốt snapshot/revision và allowlist từng tệp, dung lượng tối đa, thư mục đích; không tải toàn repo dữ liệu.
2. Kiểm tra quyền sử dụng/license từng thành phần. License thiếu của repo phương pháp chưa được giải quyết bởi license của backbone.
3. Tải từng giai đoạn; lưu URL, thời gian, size, SHA-256 và checksum nguồn nếu có. HTTP 200 hoặc filename đúng chưa đủ xác minh.
4. Không deserialize pickle không tin cậy trong môi trường nghiên cứu chính; kiểm tra an toàn cách load, keys/shapes và trạng thái missing/unexpected.
5. Môi trường riêng, không đổi các dependency đã seal. Khóa input, loss, init, controls và budget trước khi đánh giá.
6. Gate bounded: forward/backward/update, frozen invariance, original-only public input, strict reload/replay; đo RAM/VRAM/thời gian rồi mới xin duyệt lượt quality.

Đề nghị duyệt theo từng giai đoạn; việc duyệt tải không đồng nghĩa duyệt thuê GPU, cài môi trường, gửi dữ liệu lên cloud hay chạy huấn luyện. Chưa có chi phí thuê hoặc ETA training đáng tin cậy trước benchmark.
