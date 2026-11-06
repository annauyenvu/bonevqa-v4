# B + visual local tokens, no auxiliary loss — kết luận cuối

Ngày29/10/2026, ICT. Đây là nghiên cứu phát triển có ngân sách hữu hạn; không phải xác nhận lâm sàng độc lập.

## Quyết định

**Đạt tiêu chí promotion phát triển đã định trước. Đề nghị local_real_noaux với original LAST head làm ứng viên kế tiếp của B.** Giữ nguyên B và toàn bộ checkpoint đối chứng; không ghi đè bản lịch sử. Đây không phải bảo đảm đúng hơn trên mọi ảnh, mọi tập dữ liệu hoặc mọi operating point. Không tự triển khai production, không mở lại demo đã được người dùng yêu cầu tắt và không train thêm để tăng kết quả.

## Thiết kế và kiểm chứng

Hoàn tất11qualityarms: B/local trên5fold FracAtlas và placebo fold0, seed42, hai epoch/fixedLAST;4013ảnh/17998QA outer. Detector riêng TRAIN từngfold, không GT ROI ở public inference, score0.05/top3 cố định. Local route thêm tối đa3crops224px, pool2x2, question-conditioned attention/residual gate; auxiliary loss bằng0. Các loss baseline khác giữ nguyên. Local tăng computation/capacity và dropout draws, nên matchedupdates không phải matchedFLOPs.

Tất cả11qualityfreshreplays exact và frozen invariance passed. B mới cả5fold khớp bitwise toàn bộ outerrawrecords với baseline lịch sử; không cần thay đổi comparator. Saved-output independent stdlib audits cả5fold passed. Hoàn tất10headrefits TRAIN-only, mỗihead đúng10epochs, detachedfeatures, batch64; publicreplays theo role-specificbatch4 đều exact. Calibration thresholds chỉ dùng negatives của calibration; không fit từ outertest. Bootstrap2000pairedcomponent draws, seed20261105, giữ model/head/threshold cố định và resample components trong fold. Đủ4013ảnh/17998QA, thứ tự và targets khớp giữa các nhánh.

## Kết quả chính: original head, raw ngưỡng0.5

| Chỉ số | B | Local-noaux |
|---|---:|---:|
| Fracture AP |0.716275|0.749347|
| AUROC |0.883988|0.892258|
| Độ nhạy |55.94%|63.78%|
| Độ đặc hiệu |94.57%|95.57%|
| Balanced accuracy |75.26%|79.67%|
| QA exact toàn bộ |90.00%|90.73%|
| CLOSED exact |90.91%|91.94%|
| OPEN exact |88.85%|89.20%|
| Brier |0.095623|0.086896|
| TP/FN/TN/FP |400/315/3119/179|456/259/3152/146|

Primary pooledAPdelta=+0.033073, CI95%[+0.015658,+0.052056], hoàn toàn dương. Ở cùng ngưỡng0.5: giảm ròng56false negatives và33false positives trên4013ảnh, QA tăng131/17998câu (~0.7286điểm phần trăm). Đây là netcounts, không đồng nghĩa không có trường hợp từng đúng bị sai. AP tăng cả5fold, nhưng fold2 QA giảm7/3597câu, specificity giảm1.51pp; hiệu ứng không đồng đều.

## Operating point: mục tiêu calibration specificity0.90

Originalhead B→local: sensitivity66.01%→75.52%, specificity91.33%→89.57%, BA78.67%→82.55%, QA89.80%→90.09%. BA delta+3.8759pp, CI95%[+2.2966,+5.4542]pp. Sensitivitydelta+9.5105pp, CI95%[+6.5769,+12.4659]pp. Specificitydelta−1.7586pp, CI95%[−2.7847,−0.7587]pp.

Vì vậy đây là đánh đổi thật: độ nhạy tăng nhưng báo động giả tăng ở operating point này. Target0.90 là empiricalcalibrationtarget, không bảo đảm outerspecificity>=90%. Rule đã khóa yêu cầu pointdrop không quá2pp, không phải CI noninferioritymargin2pp; pointdrop đạt nhưng CI không loại trừ giảm quá2pp. Không được diễn giải thành đã chứng minh specificity không kém hơn trong margin2pp. Không điều chỉnh ngưỡng sau xem outer để ép đạt90%.

Promotionrulepassed=true vì AP CIlower>0, operatingBA CIlower>0, sensitivitydelta>0, outer specificitypointdrop≤2pp và rawoverallQA không giảm quá0.5pp. Tất cả là tiêu chí quyết định phát triển, không clinicalapproval.

## Secondary head-refit

B+head rawAP0.726574; local+head0.765130. Head-refit matched local−B AP CI95%[+0.020887,+0.056767]; operatingBA CI95%[+2.6996,+5.9156]pp. Local+head raw sensitivity76.92%,specificity89.39%,BA83.16%,QA90.12%; operating sensitivity76.92%,specificity89.69%,BA83.31%,QA90.17%.

Không tự chọn local+head làm mặc định chỉ vì AP lớn nhất: đây là secondary; raw specificity thấp hơn originallocal khoảng6.19pp và QA thấp hơn0.61pp. Head-refit thích hợp như candidate theo nhu cầu sensitivity với tradeoff được công bố, không chứng minh toàn bộ hơn originalhead. Bootstrapheadcontrast là local+head so B+head, không phải CI của headon−headoff. Không suy ra significance cho contrast chưa tính.

## Placebo và cơ chế

Fold0 real/placebo AP0.739861/0.731343, BA78.86%/77.63%; real cao hơn nhẹ nhưng QA90.06%/90.25%, placebo cao hơn. Placebo cũng tăng mạnh soB. Vì vậy cấu hình localroute có bằng chứng hiệu năng, nhưng chưa thể quy toàn bộ lợi ích cho định vị đúng tổn thương. Extra views/capacity/computation/dropout có thể đóng góp. Chỉ mộtfoldplacebo, không CI semanticcontrast và không multiseed. Detectorproposal không phải maskSAM hoặc ground-truth fracturelocalization.

## Lỗi kỹ thuật, bảo toàn và tài nguyên

V2headassessment dừng khi canonical-onlybatch lệch mixed-QA probability tối đa1.1921e-7; diagnostic thực không đổi câu trả lời. V3 khôi phục mixedbatch cho modelval nhưng áp sai cho calibration, nên dừng. V4 khôi phục chính xác rolecontract: mixedQA4modelval/test, canonical4calibration, chọn canonical AFTER fullbatchheadforward. Exactequality giữ nguyên, không nới tolerance. Existing B0/local0 fittedHEAD_LAST/TRAINfeatures/fit được hashbind và reuse, không fitlại; remainingheadsfitonce. Allv1/v2/v3sources/failures/partials giữ nguyên. Một lần bindfinalizer dùng systemPython thiếuhashlib.file_digest; emptyrelease được giữ và release_v2 tạo bằng runtimePython3.11 trước seal.

Qualityexecution~30.28h wall; headrecoveryv4 từ~08:24 đến~13:13ICT29Oct, không đồng nghĩa toànthời gian là fitting. Máy có clamshellsleep19:12–21:25ICT28Oct, ảnh hưởng elapsedwall/ETA; không cộng thêm scientificepochs hoặc gia hạn72hdeadline. V4 mỗichild timeout giới hạn deadlinegốc1793378629.708738; analysiscompleted trước deadline. CPU8/offline runtimecũ, perprocessRSS20GiB/diskreserve8GiB; cuối~42GiBfree. Các nguồn đã khóa được kiểm tra hash; demo được tắt theo user, không ảnh hưởng completedscientificoutputs.

HaiPNGcomparison/foldcomparison đã được mở và kiểm tra thực tế; không clipping, labels rõ. Biểu đồ theo fold cho thấy operatingBAfold3 không tăng, không che giấu. Biểu đồ là pointestimates, không có errorbars; CIs đọc từ bảng JSON và nội dung trên. Chỉ seal sau source/replay/analysis/figurehash/finalinterpretationchecks.

## Phạm vi kết luận và bước sau

FracAtlas và các split đã được development-informed, không validation lâm sàng mới;4013components/images không phải4013patientsđộc lập. Một trainingseed, fivefold không phải fiveseeds. Bootstrapconditional không mô tả hết trainingrandomness, detectorrandomness hoặc domainshift. Class imbalance715positive/3298negative khiến accuracyđơnlẻ không đủ; AP/sensitivity/specificity/BA và confusion phải đi cùng. Mục tiêu operatingpoint được chọn riêng từngcalibrationfold, không phải ngưỡng triển khai toànhệthống. Không thể chọn một fold có test tốt nhất thành finaldeploymentmodel mà vẫn gọi độc lập.

Kết luận hợp lý: local tokens+noaux là hướng cải thiện B được hỗ trợ bằng matchedfivefolddevelopmentstudy; originalheadraw0.5 là mặc định nghiên cứu cân bằng hơn. Giữ B làm mốc, lưu các head/operatingpoints để báo cáo đánh đổi, không khẳng định model không còn bỏsót hoặc dùng ảnh đúng cơ chế chỉ từ AP. Bước nghiên cứu tiếp theo cần protocol mới cho unseenexternalcohort/multiseed, pairedsemanticcontrols, imageablation và hiệu chuẩn triển khai; không tự chạy thêm trong phạm vi study đã kết thúc.
