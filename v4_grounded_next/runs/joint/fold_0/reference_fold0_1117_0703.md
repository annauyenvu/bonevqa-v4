# Thông tin tham chiếu có giúp không? (fold 0)

Chênh lệch so với đầu ra gốc của chính mô hình đó (chỉ ảnh, câu hỏi gốc), khoảng tin cậy 95% trong ngoặc vuông.

## J_v4cf_yolojoint_btxrd

### Chẩn đoán loại u — BTXRD test (n = 269)

| Cách | Độ chính xác | Chính xác cân bằng | Lành/ác đúng | Số lớp đoán | Chênh độ chính xác |
|---|---|---|---|---|---|
| image only (plain question) | 0,550 | 0,248 | 0,818 | 4 | — |
| wording: age, sex in the question | 0,509 | 0,203 | 0,792 | 4 | −0,041 [−0,079; −0,007] |
| wording: age, sex, site in the question | 0,494 | 0,164 | 0,777 | 2 | −0,056 [−0,104; −0,011] |
| image + data prior (age + sex) | 0,554 | 0,234 | 0,810 | 4 | +0,004 [−0,038; +0,043] |
| metadata only (age + sex) | 0,446 | 0,111 | 0,829 | 1 | −0,104 [−0,164; −0,049] |
| image + data prior (site) | 0,632 | 0,380 | 0,822 | 5 | +0,082 [+0,016; +0,153] |
| metadata only (site) | 0,584 | 0,314 | 0,770 | 4 | +0,033 [−0,071; +0,141] |
| image + data prior (age + sex + site + region + view) | 0,621 | 0,365 | 0,825 | 7 | +0,071 [+0,004; +0,140] |
| metadata only (age + sex + site + region + view) | 0,591 | 0,346 | 0,770 | 6 | +0,041 [−0,057; +0,141] |
| image + KG prior (age, site edges) | 0,528 | 0,257 | 0,792 | 6 | −0,022 [−0,056; +0,008] |
| KG prior only (age, site edges) | 0,182 | 0,183 | 0,740 | 7 | −0,368 [−0,519; −0,207] |
| majority class (no image, no metadata) | 0,446 | 0,111 | 0,829 | 1 | −0,104 [−0,164; −0,046] |

### Chẩn đoán loại u — BTXRD external (n = 522)

| Cách | Độ chính xác | Chính xác cân bằng | Lành/ác đúng | Số lớp đoán | Chênh độ chính xác |
|---|---|---|---|---|---|
| image only (plain question) | 0,287 | 0,161 | 0,665 | 5 | — |
| wording: age, sex in the question | 0,270 | 0,152 | 0,636 | 4 | −0,017 [−0,038; +0,004] |
| wording: age, sex, site in the question | 0,268 | 0,146 | 0,657 | 3 | −0,019 [−0,046; +0,008] |
| image + data prior (age + sex) | 0,305 | 0,174 | 0,699 | 6 | +0,017 [−0,006; +0,042] |
| metadata only (age + sex) | 0,239 | 0,111 | 0,795 | 1 | −0,048 [−0,099; +0,002] |
| image + data prior (site) | 0,375 | 0,217 | 0,686 | 5 | +0,088 [+0,051; +0,126] |
| metadata only (site) | 0,345 | 0,237 | 0,682 | 4 | +0,057 [−0,006; +0,119] |
| image + data prior (age + sex + site + region + view) | 0,387 | 0,223 | 0,709 | 5 | +0,100 [+0,061; +0,141] |
| metadata only (age + sex + site + region + view) | 0,356 | 0,231 | 0,720 | 7 | +0,069 [+0,004; +0,131] |
| image + KG prior (age, site edges) | 0,287 | 0,162 | 0,667 | 5 | +0,000 [−0,023; +0,023] |
| KG prior only (age, site edges) | 0,230 | 0,240 | 0,793 | 7 | −0,057 [−0,155; +0,042] |
| majority class (no image, no metadata) | 0,239 | 0,111 | 0,795 | 1 | −0,048 [−0,101; +0,004] |

### Lành hay ác — BTXRD test (n = 269, dương 46)

| Cách | AUPRC | AUROC | Chênh AUROC |
|---|---|---|---|
| image only (plain question) | 0,574 | 0,833 | — |
| wording: age, sex in the question | 0,550 | 0,828 | −0,005 [−0,023; +0,016] |
| wording: age, sex, site in the question | 0,137 | 0,407 | −0,425 [−0,555; −0,287] |
| image + data prior (age + sex) | 0,574 | 0,833 | +0,000 [+0,000; +0,000] |
| metadata only (age + sex) | 0,167 | 0,474 | −0,359 [−0,525; −0,202] |
| image + data prior (site) | 0,595 | 0,861 | +0,029 [+0,009; +0,049] |
| metadata only (site) | 0,320 | 0,762 | −0,070 [−0,177; +0,029] |
| image + data prior (age + sex + site + region + view) | 0,578 | 0,849 | +0,016 [+0,004; +0,028] |
| metadata only (age + sex + site + region + view) | 0,335 | 0,776 | −0,057 [−0,163; +0,047] |
| proposal flag only | 0,170 | 0,497 | −0,335 [−0,393; −0,277] |

### Lành hay ác — BTXRD external (n = 522, dương 107)

| Cách | AUPRC | AUROC | Chênh AUROC |
|---|---|---|---|
| image only (plain question) | 0,385 | 0,610 | — |
| wording: age, sex in the question | 0,380 | 0,607 | −0,003 [−0,016; +0,009] |
| wording: age, sex, site in the question | 0,173 | 0,427 | −0,182 [−0,298; −0,061] |
| image + data prior (age + sex) | 0,385 | 0,610 | +0,000 [+0,000; +0,000] |
| metadata only (age + sex) | 0,254 | 0,572 | −0,038 [−0,162; +0,088] |
| image + data prior (site) | 0,397 | 0,636 | +0,026 [+0,013; +0,042] |
| metadata only (site) | 0,254 | 0,620 | +0,010 [−0,087; +0,109] |
| image + data prior (age + sex + site + region + view) | 0,387 | 0,626 | +0,017 [+0,006; +0,027] |
| metadata only (age + sex + site + region + view) | 0,314 | 0,642 | +0,032 [−0,071; +0,138] |
| proposal flag only | 0,200 | 0,486 | −0,124 [−0,190; −0,048] |

### Có u không — BTXRD test (n = 587, dương 269)

| Cách | AUPRC | AUROC | Chênh AUROC |
|---|---|---|---|
| image only (plain question) | 0,926 | 0,934 | — |
| wording: age, sex in the question | 0,924 | 0,933 | −0,001 [−0,004; +0,003] |
| image + data prior (age + sex) | 0,933 | 0,935 | +0,001 [−0,006; +0,007] |
| metadata only (age + sex) | 0,574 | 0,643 | −0,290 [−0,375; −0,210] |
| image + data prior (age + sex + region + view) | 0,935 | 0,936 | +0,002 [−0,003; +0,007] |
| metadata only (age + sex + region + view) | 0,622 | 0,671 | −0,263 [−0,336; −0,190] |
| proposal flag only | 0,729 | 0,828 | −0,105 [−0,130; −0,082] |

### Có u không — BTXRD external (n = 808, dương 522)

| Cách | AUPRC | AUROC | Chênh AUROC |
|---|---|---|---|
| image only (plain question) | 0,906 | 0,842 | — |
| wording: age, sex in the question | 0,905 | 0,837 | −0,005 [−0,010; −0,000] |
| image + data prior (age + sex) | 0,905 | 0,845 | +0,003 [−0,009; +0,014] |
| metadata only (age + sex) | 0,745 | 0,622 | −0,220 [−0,287; −0,154] |
| image + data prior (age + sex + region + view) | 0,906 | 0,846 | +0,004 [−0,005; +0,013] |
| metadata only (age + sex + region + view) | 0,762 | 0,666 | −0,176 [−0,245; −0,113] |
| proposal flag only | 0,800 | 0,763 | −0,079 [−0,104; −0,055] |

### Gãy xương — FracAtlas test (n = 803, gãy 143)

| Cách | AUPRC | AUROC | Chênh AUPRC |
|---|---|---|---|
| image only | 0,799 | 0,931 | — |
| image + data prior (region + view) | 0,799 | 0,931 | +0,000 [+0,000; +0,000] |
| metadata only (region + view) | 0,312 | 0,703 | −0,487 [−0,551; −0,405] |
| image + data prior (region + view + hardware) | 0,799 | 0,931 | +0,000 [+0,000; +0,000] |
| metadata only (region + view + hardware) | 0,421 | 0,742 | −0,378 [−0,452; −0,301] |
| image + data prior (hardware) | 0,799 | 0,931 | +0,001 [−0,000; +0,003] |
| metadata only (hardware) | 0,259 | 0,549 | −0,540 [−0,596; −0,482] |

## K_v4cf_kg

### Chẩn đoán loại u — BTXRD test (n = 269)

| Cách | Độ chính xác | Chính xác cân bằng | Lành/ác đúng | Số lớp đoán | Chênh độ chính xác |
|---|---|---|---|---|---|
| image only (plain question) | 0,546 | 0,243 | 0,807 | 4 | — |
| wording: age, sex in the question | 0,535 | 0,225 | 0,803 | 4 | −0,011 [−0,038; +0,011] |
| wording: age, sex, site in the question | 0,517 | 0,171 | 0,818 | 2 | −0,030 [−0,068; +0,004] |
| image + data prior (age + sex) | 0,550 | 0,254 | 0,810 | 5 | +0,004 [−0,040; +0,043] |
| metadata only (age + sex) | 0,446 | 0,111 | 0,829 | 1 | −0,100 [−0,161; −0,043] |
| image + data prior (site) | 0,621 | 0,353 | 0,818 | 5 | +0,074 [+0,008; +0,144] |
| metadata only (site) | 0,584 | 0,314 | 0,770 | 4 | +0,037 [−0,065; +0,142] |
| image + data prior (age + sex + site + region + view) | 0,621 | 0,361 | 0,810 | 6 | +0,074 [+0,008; +0,147] |
| metadata only (age + sex + site + region + view) | 0,591 | 0,346 | 0,770 | 6 | +0,045 [−0,055; +0,147] |
| image + KG prior (age, site edges) | 0,535 | 0,269 | 0,792 | 6 | −0,011 [−0,051; +0,026] |
| KG prior only (age, site edges) | 0,182 | 0,183 | 0,740 | 7 | −0,364 [−0,523; −0,208] |
| majority class (no image, no metadata) | 0,446 | 0,111 | 0,829 | 1 | −0,100 [−0,161; −0,044] |

### Chẩn đoán loại u — BTXRD external (n = 522)

| Cách | Độ chính xác | Chính xác cân bằng | Lành/ác đúng | Số lớp đoán | Chênh độ chính xác |
|---|---|---|---|---|---|
| image only (plain question) | 0,272 | 0,149 | 0,669 | 5 | — |
| wording: age, sex in the question | 0,285 | 0,159 | 0,651 | 4 | +0,013 [−0,004; +0,033] |
| wording: age, sex, site in the question | 0,264 | 0,139 | 0,715 | 2 | −0,008 [−0,032; +0,017] |
| image + data prior (age + sex) | 0,301 | 0,169 | 0,716 | 5 | +0,029 [+0,005; +0,053] |
| metadata only (age + sex) | 0,239 | 0,111 | 0,795 | 1 | −0,033 [−0,080; +0,012] |
| image + data prior (site) | 0,354 | 0,201 | 0,680 | 5 | +0,082 [+0,048; +0,120] |
| metadata only (site) | 0,345 | 0,237 | 0,682 | 4 | +0,073 [+0,016; +0,131] |
| image + data prior (age + sex + site + region + view) | 0,370 | 0,212 | 0,693 | 5 | +0,098 [+0,061; +0,136] |
| metadata only (age + sex + site + region + view) | 0,356 | 0,231 | 0,720 | 7 | +0,084 [+0,023; +0,141] |
| image + KG prior (age, site edges) | 0,274 | 0,151 | 0,676 | 5 | +0,002 [−0,016; +0,020] |
| KG prior only (age, site edges) | 0,230 | 0,240 | 0,793 | 7 | −0,042 [−0,133; +0,044] |
| majority class (no image, no metadata) | 0,239 | 0,111 | 0,795 | 1 | −0,033 [−0,083; +0,014] |

### Lành hay ác — BTXRD test (n = 269, dương 46)

| Cách | AUPRC | AUROC | Chênh AUROC |
|---|---|---|---|
| image only (plain question) | 0,565 | 0,825 | — |
| wording: age, sex in the question | 0,537 | 0,829 | +0,004 [−0,008; +0,018] |
| wording: age, sex, site in the question | 0,258 | 0,699 | −0,126 [−0,187; −0,062] |
| image + data prior (age + sex) | 0,565 | 0,825 | +0,000 [+0,000; +0,000] |
| metadata only (age + sex) | 0,167 | 0,474 | −0,351 [−0,500; −0,200] |
| image + data prior (site) | 0,594 | 0,856 | +0,031 [+0,011; +0,052] |
| metadata only (site) | 0,320 | 0,762 | −0,062 [−0,168; +0,041] |
| image + data prior (age + sex + site + region + view) | 0,573 | 0,835 | +0,010 [+0,004; +0,017] |
| metadata only (age + sex + site + region + view) | 0,335 | 0,776 | −0,049 [−0,164; +0,057] |
| proposal flag only | 0,170 | 0,497 | −0,327 [−0,385; −0,265] |

### Lành hay ác — BTXRD external (n = 522, dương 107)

| Cách | AUPRC | AUROC | Chênh AUROC |
|---|---|---|---|
| image only (plain question) | 0,365 | 0,594 | — |
| wording: age, sex in the question | 0,339 | 0,586 | −0,007 [−0,016; +0,001] |
| wording: age, sex, site in the question | 0,194 | 0,507 | −0,087 [−0,128; −0,046] |
| image + data prior (age + sex) | 0,365 | 0,594 | +0,000 [+0,000; +0,000] |
| metadata only (age + sex) | 0,254 | 0,572 | −0,022 [−0,154; +0,110] |
| image + data prior (site) | 0,382 | 0,622 | +0,029 [+0,013; +0,045] |
| metadata only (site) | 0,254 | 0,620 | +0,026 [−0,060; +0,119] |
| image + data prior (age + sex + site + region + view) | 0,367 | 0,601 | +0,008 [+0,003; +0,013] |
| metadata only (age + sex + site + region + view) | 0,314 | 0,642 | +0,048 [−0,055; +0,153] |
| proposal flag only | 0,200 | 0,486 | −0,108 [−0,171; −0,038] |

### Có u không — BTXRD test (n = 587, dương 269)

| Cách | AUPRC | AUROC | Chênh AUROC |
|---|---|---|---|
| image only (plain question) | 0,932 | 0,934 | — |
| wording: age, sex in the question | 0,932 | 0,934 | +0,000 [−0,002; +0,003] |
| image + data prior (age + sex) | 0,932 | 0,934 | −0,000 [−0,006; +0,006] |
| metadata only (age + sex) | 0,574 | 0,643 | −0,290 [−0,372; −0,210] |
| image + data prior (age + sex + region + view) | 0,936 | 0,936 | +0,002 [−0,003; +0,007] |
| metadata only (age + sex + region + view) | 0,622 | 0,671 | −0,262 [−0,335; −0,195] |
| proposal flag only | 0,729 | 0,828 | −0,105 [−0,130; −0,082] |

### Có u không — BTXRD external (n = 808, dương 522)

| Cách | AUPRC | AUROC | Chênh AUROC |
|---|---|---|---|
| image only (plain question) | 0,895 | 0,840 | — |
| wording: age, sex in the question | 0,893 | 0,837 | −0,003 [−0,006; −0,001] |
| image + data prior (age + sex) | 0,899 | 0,843 | +0,002 [−0,009; +0,013] |
| metadata only (age + sex) | 0,745 | 0,622 | −0,218 [−0,289; −0,152] |
| image + data prior (age + sex + region + view) | 0,901 | 0,846 | +0,005 [−0,004; +0,014] |
| metadata only (age + sex + region + view) | 0,762 | 0,666 | −0,175 [−0,240; −0,112] |
| proposal flag only | 0,800 | 0,763 | −0,078 [−0,101; −0,053] |

### Gãy xương — FracAtlas test (n = 803, gãy 143)

| Cách | AUPRC | AUROC | Chênh AUPRC |
|---|---|---|---|
| image only | 0,814 | 0,934 | — |
| image + data prior (region + view) | 0,814 | 0,934 | +0,000 [+0,000; +0,000] |
| metadata only (region + view) | 0,312 | 0,703 | −0,502 [−0,568; −0,422] |
| image + data prior (region + view + hardware) | 0,814 | 0,934 | +0,000 [+0,000; +0,000] |
| metadata only (region + view + hardware) | 0,421 | 0,742 | −0,393 [−0,465; −0,317] |
| image + data prior (hardware) | 0,814 | 0,934 | +0,000 [−0,000; +0,001] |
| metadata only (hardware) | 0,259 | 0,549 | −0,555 [−0,611; −0,494] |

## K0_v4cf_kgempty

### Chẩn đoán loại u — BTXRD test (n = 269)

| Cách | Độ chính xác | Chính xác cân bằng | Lành/ác đúng | Số lớp đoán | Chênh độ chính xác |
|---|---|---|---|---|---|
| image only (plain question) | 0,543 | 0,227 | 0,818 | 4 | — |
| wording: age, sex in the question | 0,535 | 0,215 | 0,822 | 4 | −0,007 [−0,022; +0,007] |
| wording: age, sex, site in the question | 0,513 | 0,172 | 0,788 | 2 | −0,030 [−0,066; +0,007] |
| image + data prior (age + sex) | 0,546 | 0,236 | 0,818 | 4 | +0,004 [−0,036; +0,042] |
| metadata only (age + sex) | 0,446 | 0,111 | 0,829 | 1 | −0,097 [−0,151; −0,048] |
| image + data prior (site) | 0,613 | 0,332 | 0,822 | 5 | +0,071 [+0,008; +0,132] |
| metadata only (site) | 0,584 | 0,314 | 0,770 | 4 | +0,041 [−0,063; +0,144] |
| image + data prior (age + sex + site + region + view) | 0,617 | 0,341 | 0,825 | 6 | +0,074 [+0,008; +0,142] |
| metadata only (age + sex + site + region + view) | 0,591 | 0,346 | 0,770 | 6 | +0,048 [−0,048; +0,151] |
| image + KG prior (age, site edges) | 0,546 | 0,244 | 0,822 | 4 | +0,004 [−0,028; +0,036] |
| KG prior only (age, site edges) | 0,182 | 0,183 | 0,740 | 7 | −0,361 [−0,521; −0,201] |
| majority class (no image, no metadata) | 0,446 | 0,111 | 0,829 | 1 | −0,097 [−0,150; −0,048] |

### Chẩn đoán loại u — BTXRD external (n = 522)

| Cách | Độ chính xác | Chính xác cân bằng | Lành/ác đúng | Số lớp đoán | Chênh độ chính xác |
|---|---|---|---|---|---|
| image only (plain question) | 0,276 | 0,150 | 0,695 | 5 | — |
| wording: age, sex in the question | 0,274 | 0,151 | 0,669 | 4 | −0,002 [−0,018; +0,013] |
| wording: age, sex, site in the question | 0,272 | 0,144 | 0,699 | 2 | −0,004 [−0,032; +0,023] |
| image + data prior (age + sex) | 0,297 | 0,166 | 0,716 | 5 | +0,021 [−0,002; +0,046] |
| metadata only (age + sex) | 0,239 | 0,111 | 0,795 | 1 | −0,036 [−0,082; +0,008] |
| image + data prior (site) | 0,366 | 0,207 | 0,697 | 5 | +0,090 [+0,056; +0,126] |
| metadata only (site) | 0,345 | 0,237 | 0,682 | 4 | +0,069 [+0,011; +0,127] |
| image + data prior (age + sex + site + region + view) | 0,364 | 0,207 | 0,711 | 5 | +0,088 [+0,052; +0,127] |
| metadata only (age + sex + site + region + view) | 0,356 | 0,231 | 0,720 | 7 | +0,080 [+0,016; +0,140] |
| image + KG prior (age, site edges) | 0,285 | 0,157 | 0,701 | 5 | +0,010 [−0,013; +0,032] |
| KG prior only (age, site edges) | 0,230 | 0,240 | 0,793 | 7 | −0,046 [−0,137; +0,046] |
| majority class (no image, no metadata) | 0,239 | 0,111 | 0,795 | 1 | −0,036 [−0,083; +0,006] |

### Lành hay ác — BTXRD test (n = 269, dương 46)

| Cách | AUPRC | AUROC | Chênh AUROC |
|---|---|---|---|
| image only (plain question) | 0,579 | 0,829 | — |
| wording: age, sex in the question | 0,543 | 0,834 | +0,005 [−0,011; +0,021] |
| wording: age, sex, site in the question | 0,412 | 0,793 | −0,036 [−0,090; +0,023] |
| image + data prior (age + sex) | 0,579 | 0,829 | +0,000 [+0,000; +0,000] |
| metadata only (age + sex) | 0,167 | 0,474 | −0,355 [−0,517; −0,193] |
| image + data prior (site) | 0,621 | 0,859 | +0,030 [+0,007; +0,053] |
| metadata only (site) | 0,320 | 0,762 | −0,067 [−0,171; +0,037] |
| image + data prior (age + sex + site + region + view) | 0,590 | 0,838 | +0,009 [+0,002; +0,016] |
| metadata only (age + sex + site + region + view) | 0,335 | 0,776 | −0,054 [−0,169; +0,051] |
| proposal flag only | 0,170 | 0,497 | −0,332 [−0,393; −0,265] |

### Lành hay ác — BTXRD external (n = 522, dương 107)

| Cách | AUPRC | AUROC | Chênh AUROC |
|---|---|---|---|
| image only (plain question) | 0,377 | 0,605 | — |
| wording: age, sex in the question | 0,360 | 0,612 | +0,006 [−0,004; +0,016] |
| wording: age, sex, site in the question | 0,213 | 0,532 | −0,073 [−0,110; −0,037] |
| image + data prior (age + sex) | 0,377 | 0,605 | +0,000 [+0,000; +0,000] |
| metadata only (age + sex) | 0,254 | 0,572 | −0,034 [−0,155; +0,095] |
| image + data prior (site) | 0,405 | 0,634 | +0,028 [+0,014; +0,045] |
| metadata only (site) | 0,254 | 0,620 | +0,014 [−0,076; +0,110] |
| image + data prior (age + sex + site + region + view) | 0,386 | 0,613 | +0,007 [+0,003; +0,012] |
| metadata only (age + sex + site + region + view) | 0,314 | 0,642 | +0,036 [−0,059; +0,133] |
| proposal flag only | 0,200 | 0,486 | −0,120 [−0,183; −0,050] |

### Có u không — BTXRD test (n = 587, dương 269)

| Cách | AUPRC | AUROC | Chênh AUROC |
|---|---|---|---|
| image only (plain question) | 0,936 | 0,937 | — |
| wording: age, sex in the question | 0,927 | 0,928 | −0,009 [−0,016; −0,003] |
| image + data prior (age + sex) | 0,937 | 0,937 | +0,000 [−0,005; +0,006] |
| metadata only (age + sex) | 0,574 | 0,643 | −0,293 [−0,377; −0,213] |
| image + data prior (age + sex + region + view) | 0,939 | 0,939 | +0,002 [−0,003; +0,007] |
| metadata only (age + sex + region + view) | 0,622 | 0,671 | −0,266 [−0,343; −0,197] |
| proposal flag only | 0,729 | 0,828 | −0,108 [−0,133; −0,084] |

### Có u không — BTXRD external (n = 808, dương 522)

| Cách | AUPRC | AUROC | Chênh AUROC |
|---|---|---|---|
| image only (plain question) | 0,903 | 0,838 | — |
| wording: age, sex in the question | 0,902 | 0,833 | −0,005 [−0,013; +0,004] |
| image + data prior (age + sex) | 0,906 | 0,843 | +0,005 [−0,007; +0,015] |
| metadata only (age + sex) | 0,745 | 0,622 | −0,215 [−0,283; −0,147] |
| image + data prior (age + sex + region + view) | 0,908 | 0,844 | +0,006 [−0,003; +0,015] |
| metadata only (age + sex + region + view) | 0,762 | 0,666 | −0,172 [−0,241; −0,109] |
| proposal flag only | 0,800 | 0,763 | −0,075 [−0,099; −0,051] |

### Gãy xương — FracAtlas test (n = 803, gãy 143)

| Cách | AUPRC | AUROC | Chênh AUPRC |
|---|---|---|---|
| image only | 0,800 | 0,929 | — |
| image + data prior (region + view) | 0,800 | 0,929 | +0,000 [+0,000; +0,000] |
| metadata only (region + view) | 0,312 | 0,703 | −0,488 [−0,552; −0,413] |
| image + data prior (region + view + hardware) | 0,800 | 0,929 | +0,000 [+0,000; +0,000] |
| metadata only (region + view + hardware) | 0,421 | 0,742 | −0,380 [−0,447; −0,304] |
| image + data prior (hardware) | 0,801 | 0,930 | +0,000 [−0,000; +0,002] |
| metadata only (hardware) | 0,259 | 0,549 | −0,542 [−0,599; −0,481] |
