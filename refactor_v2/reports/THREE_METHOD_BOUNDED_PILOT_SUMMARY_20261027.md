# Tổng kết ba pilot giới hạn: LaPA → FAVP → Lens

Đã hoàn tất tuần tự cả ba **pilot có điều chỉnh được công bố**, gồm kiểm thử mô hình thật, backward/update, fresh-process replay, kiểm định đầu ra độc lập, bootstrap, xem biểu đồ thật và đóng dấu artifact. Không còn worker/coordinator/guard/giữ thức của chương trình này đang chạy. Đây không phải tuyên bố tái lập nguyên bản100%, hoàn tất toàn bộ nghiên cứu BoneVQA, hay tạo được mô hình tốt hơn/đủ dùng lâm sàng.

## Kết quả chính

| Phần | Thử nghiệm đã đóng | Kết quả có thể kết luận |
|---|---|---|
| LaPA | 5nhánh,4epochs/292updates mỗi nhánh;1.432QA;52artifacts/197dependencies | AP carrier0,660138;GM+LF0,526043;authorPF0,430638. Tương phản chính PF−GM+LF là−0,095405, CI[−0,172551;−0,011226]. PF tái dựng không cải thiện trong cấu hình này; không nâng cấp nhánh PF. |
| FAVP | 32TRAIN/8model-val,2epochs/64updates;25artifacts/606dependencies | PRE0/8→LAST6/8, nhưng LAST trả lời no toàn bộ và bằng baseline TRAIN-majority no. Chưa có bằng chứng lợi ích điều kiện theo ảnh/local branch. |
| Lens | 28TRAIN hợp lệ từ32candidates;2nhánh cùng3views;1alignment+2QAepochs,42microbatches/12updates mỗi nhánh;44artifacts/222dependencies | CE-only và Lens-loss đều6/8 ở LAST, đều No toàn bộ, bằng majority no. Lens−CE=0; CI bootstrap[0,0] chỉ do đầu ra trên8case giống nhau, không chứng minh tương đương tổng quát. Sensitivity0, specificity1, BA0,5. Chưa thấy lợi ích từ bật DCL/InfoNCE. |

Kết quả kỹ thuật là chạy và tái lập được các route đã mô tả. Kết quả hiệu năng **chưa ủng hộ việc tự động ghép/promote các nhánh mới**. Các pilot âm được giữ nguyên; không tăng seed/epoch, đổi threshold/parser hay chọn checkpoint khác để tìm kết quả tốt hơn.

## Điểm quan trọng khi đọc Lens/FAVP

Lens TRAIN27no/1yes; FAVP TRAIN31no/1yes. Cùng8ảnh model-validation đã dùng trong phát triển có6no/2yes. Vì thế75% có thể đạt chỉ bằng trả lời no, không cần xem ảnh; cả hai pilot generative đã rơi vào trường hợp này. Lens PRE strict EM0 nhưng leading-yes/no diagnostic đã5/8 do câu đầy đủ; LAST trả lời ngắn đúng định dạng hơn không đồng nghĩa phát hiện gãy tốt hơn. Lens secondary BA giảm từ0,75 ở PRE xuống0,5 ở LAST, bỏ sót cả hai ca dương tính. Giữ primary/secondary đã khóa, không đổi tiêu chí sau kết quả.

Lens không có câu LAST bị cắt ở cap32tokens; FAVP vẫn có giới hạn tất cả8câu chạm cap20tokens. Không che giấu khác biệt này hoặc dùng75% để so trực tiếp hai phương pháp với ngân sách/backbone khác nhau. Bootstrap chỉ mô tả các component đã biết với một training seed; component không phải bệnh nhân đã xác minh,8case tái sử dụng không phải tập độc lập.

## Mức độ bám nguồn và phần chưa giải quyết

- **LaPA:** native M3AE image/text/MM đã chạy, nhưng image/text frozen, prompt/head mới, graph577/node order gốc chưa có nên dùng SLAKE409 tái dựng được người dùng chấp nhận. Endpoint41class và ngân sách ngắn không phải full original fine-tuning; không kết luận LaPA gốc thất bại.
- **FAVP:** native SAMMed2D/EVA/QFormer/Vicuna đã chạy; thiếu projection/stage1–2 handoff gốc nên projection random. Source SAM không nhận proposal trên256TRAIN+8val ở threshold không đổi; route fallback ảnh gốc khiến local-on/off cùng đầu vào, không chứng minh hiệu quả local. Các correction/adaptation LoRA, precisionCPU, generation compatibility được ghi rõ và source/failures giữ nguyên.
- **Lens:** native SAMHuge/SmolVLM và source DCL/InfoNCE/collator/pooling đã chạy. TRAIN4ảnh không có mask đủ tiêu chí bị loại không thay thế. Processor1024longedge thay2048, SAMpointbatch8 thay64, checkpointingCPU và nhãn region ảnh từ CSV thay organ-specific supervision là điều chỉnh công bố. Không fallback organ giả hoặc annotationROI inference. Hai nhánh chỉ cô lập loss route khi cùng3view training; không phải whole-view ablation hay8nhánh tích hợp. Extra visual forward của loss-on cũng khác compute/RNG, không phải thí nghiệm dropout bitwise matched.

Nguồn/asset hashes và các lỗi phiên bản cũ được giữ. Thiếu provenance/asset gốc và giới hạn phần cứng không được biến thành tuyên bố nguyên bản đã tái lập hoàn chỉnh; cũng không có chấp thuận phân phối lại từ việc tải/hash thành công.

## So với BoneVQA v3 và bước tiếp theo

Các run tham chiếu có checkpoint/report riêng, không tự thay thế hay ghép vào BoneVQA v3; chúng vẫn dùng chung dữ liệu/hạ tầng, không độc lập về bằng chứng. Mốc nativeB grouped outer trước đây (T/V/L/P tắt) có17.998QA/4.013ảnh, raw fracture BA75,26%, AP0,71627 và overall QA90,00%. Không xếp hạng các số đó trực tiếp với LaPA1.432QA/41class hoặc FAVP/Lens8case generation. Chưa có bằng chứng từ các pilot mới rằng một phiên bản tích hợp tốt hơn v3.

Có thể thiết kế ablation bật/tắt sau này, nhưng cần một backbone/cohort/khởi tạo/ngân sách/endpoint được khóa chung; phân biệt huấn luyện với module tắt/bật và chỉ tắt tại inference. Lens chủ yếu thay đổi loss train, không có cùng một nút inference để tắt. Không tự mở8nhánh training, tải thêm tài nguyên hay kéo dài pilot từ báo cáo này. Chương trình theo dõi30phút của ba pilot được kết thúc sau bàn giao vì mọi tác vụ thuộc giai đoạn này đã đóng; không dừng ở một tiểu bước còn chạy.

## Tài nguyên và bằng chứng

Lens dùng95,92phút theo timer thực thi/coordinator và161,04phút elapsed thực gồm giấc ngủ máy. Clamshell sleep18:22:05→wake19:30:05 được công bố; guard riêng bảo toàn222phụ thuộc và giữ nguyên deadline4h, hoàn tất lúc khoảng20:49ICT trước hạn22:08ICT. Không restart training hay gia hạn để bù sleep.

- [LaPA: diễn giải cuối](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/lapa_native_quality_v1/FINAL_INTERPRETATION.md)
- [FAVP: diễn giải cuối](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/favp_bounded_quality_v1/FINAL_INTERPRETATION.md)
- [Lens: diễn giải cuối](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/lens_bounded_quality_v1/FINAL_INTERPRETATION.md)
- [Lens: closure v2](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/lens_bounded_quality_v1/closure_seal.json)
- [Phạm vi so sánh với v3](/Users/uyen/Downloads/BONEVQA/refactor_v2/reports/THREE_METHOD_COMPARISON_SCOPE_V1.md)
