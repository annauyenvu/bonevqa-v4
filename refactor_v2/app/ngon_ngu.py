import re
import unicodedata

VUNG_VIET_ANH = [
    (r"ban tay|co tay|ngon tay|khop tay", "hand"),
    (r"cang chan|ban chan|co chan|chan", "leg"),
    (r"khop hang|xuong chau|hang", "hip"),
    (r"khop vai|xuong don|vai", "shoulder"),
    (r"tay", "hand"),
]

VUNG_ANH_VIET = {
    "hand": "bàn tay", "leg": "chân", "hip": "háng", "shoulder": "vai",
    "unknown": "không xác định", "body": "cơ thể", "mixed": "nhiều vùng",
}

VI_TRI_ANH_VIET = {
    "upper": "phía trên", "lower": "phía dưới", "middle": "ở giữa", "center": "trung tâm",
    "left": "bên trái", "right": "bên phải",
    "upper-left": "trên bên trái", "upper-right": "trên bên phải", "upper-center": "trên trung tâm",
    "lower-left": "dưới bên trái", "lower-right": "dưới bên phải", "lower-center": "dưới trung tâm",
    "middle-left": "giữa bên trái", "middle-right": "giữa bên phải",
}

SO_ANH_VIET = {
    "zero": "không có ổ gãy nào", "one": "một", "two": "hai", "three": "ba",
    "four": "bốn", "five": "năm", "six": "sáu",
}

SO_TRONG_CAU = {"zero": "không", "one": "một", "two": "hai", "three": "ba", "four": "bốn", "five": "năm", "six": "sáu"}

DAP_AN_ANH_VIET = {
    "yes": "có", "no": "không",
    "fracture": "có gãy xương", "no abnormality": "không có bất thường",
    "x-ray": "ảnh X-quang", "xray": "ảnh X-quang",
    "frontal": "tư thế thẳng", "lateral": "tư thế nghiêng", "oblique": "tư thế chếch",
}

TU_GAY = r"gay|nut|ran|fracture"

MAU_CAU_HOI = [
    (TU_GAY, r"bao nhieu|\bmay\b|so luong|dem", "How many fracture sites are visible?"),
    (TU_GAY, r"o dau|vi tri nao|cho nao|nam dau|dau\?*$", "Where is the fracture located?"),
    (TU_GAY, r"nhieu|da o|hon mot", "Are there multiple fractures?"),
    (r"kim loai|\bnep\b|\bvit\b|\bdinh\b|dung cu|implant|hardware", r"", "Is there any orthopedic hardware or implant visible?"),
    (r"mo ta|nhan xet|thay gi|ket luan|doc ket qua|nhin thay", r"", "Describe the finding."),
    (r"bat thuong|ton thuong|benh ly|abnormal", r"", "What abnormality is seen in this X-ray?"),
    (r"loai anh|ky thuat|phuong thuc|anh gi|chup gi|modality", r"", "What type of imaging is this?"),
    (r"vung co the|bo phan|vung nao|phan nao|chi nao|vi tri co the|\bvung\b.*\bnao\b|\bvung\b.*\bgi\b", r"", "Which body region is shown in this X-ray?"),
    (TU_GAY, r"", "Is there a fracture in this X-ray?"),
]

DAU_HIEU_DONG = (r"\bco\b.*\bkhong\b", r"phai khong", r"dung khong", r"hay khong")

MO_DAU_TIENG_ANH = (
    "is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ", "could ", "has ", "have ", "had ",
    "should ", "will ", "would ", "what ", "which ", "where ", "when ", "who ", "why ", "how ",
    "describe", "list ", "name ", "identify",
)

TU_TIENG_VIET = (
    r"\bkhong\b", r"\bco\b", r"\bla\b", r"\bnao\b", r"\bgi\b", r"\bdau\b", r"bao nhieu",
    r"the nao", r"\banh\b", r"xuong", r"\bgay\b", r"\bvung\b", r"bo phan", r"mo ta", r"\bnay\b", r"\bcua\b",
)


def bo_dau(text):
    nfkd = unicodedata.normalize("NFD", (text or "").replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn").lower()


def la_tieng_viet(text):
    thuong = (text or "").strip().lower()
    if not thuong:
        return False
    if any(ch in thuong for ch in "ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"):
        return True
    if thuong.startswith(MO_DAU_TIENG_ANH):
        return False
    khong_dau = bo_dau(thuong)
    return any(re.search(p, khong_dau) for p in TU_TIENG_VIET)


def _vung_trong_cau(khong_dau):
    for mau, vung in VUNG_VIET_ANH:
        if re.search(rf"\b(?:{mau})\b", khong_dau):
            return vung
    return None


def cau_hoi_dong_tieng_viet(question):
    khong_dau = bo_dau((question or "").strip())
    return any(re.search(p, khong_dau) for p in DAU_HIEU_DONG)


def sang_tieng_anh(question):
    cau = (question or "").strip()
    if not cau or not la_tieng_viet(cau):
        return cau, False
    khong_dau = bo_dau(cau)
    vung = _vung_trong_cau(khong_dau)
    la_cau_dong = any(re.search(p, khong_dau) for p in DAU_HIEU_DONG)
    if vung and re.search(TU_GAY, khong_dau) and la_cau_dong:
        return f"Is the {vung} fractured?", True
    if vung and re.search(r"x.?quang|hinh anh|\banh\b", khong_dau) and la_cau_dong:
        return f"Is this an X-ray of the {vung}?", True
    for tu_khoa, phu, mau in MAU_CAU_HOI:
        if re.search(tu_khoa, khong_dau) and (not phu or re.search(phu, khong_dau)):
            return mau, True
    if vung:
        return f"Is this an X-ray of the {vung}?", True
    return cau, False


def _dich_vung(text):
    ket_qua = text
    for anh, viet in sorted(VUNG_ANH_VIET.items(), key=lambda kv: -len(kv[0])):
        ket_qua = re.sub(rf"\b{re.escape(anh)}\b", viet, ket_qua)
    return ket_qua.replace(" and ", " và ")


def _dich_vi_tri(text):
    ket_qua = text
    for anh, viet in sorted(VI_TRI_ANH_VIET.items(), key=lambda kv: -len(kv[0])):
        ket_qua = re.sub(rf"\b{re.escape(anh)}\b", viet, ket_qua)
    return ket_qua.replace(" and ", " và ")


def sang_tieng_viet(answer):
    goc = (answer or "").strip()
    if not goc:
        return goc
    chuan = goc.lower().rstrip(".").strip()
    if chuan in DAP_AN_ANH_VIET:
        return DAP_AN_ANH_VIET[chuan]
    if chuan in SO_ANH_VIET:
        return SO_ANH_VIET[chuan]
    if chuan in VUNG_ANH_VIET:
        return VUNG_ANH_VIET[chuan]

    m = re.fullmatch(r"no fracture is seen in the (.+)", chuan)
    if m:
        return f"Không thấy gãy xương ở {_dich_vung(m.group(1))}."
    m = re.fullmatch(r"a fracture is seen in the (.+?)(?: at the (.+))?", chuan)
    if m:
        cau = f"Thấy một ổ gãy xương ở {_dich_vung(m.group(1))}"
        if m.group(2):
            cau += f", vị trí {_dich_vi_tri(m.group(2))}"
        return cau + "."
    m = re.fullmatch(r"(\w+) fractures are seen in the (.+?)(?: at the (.+))?", chuan)
    if m:
        so = SO_TRONG_CAU.get(m.group(1).lower(), m.group(1))
        cau = f"Thấy {so} ổ gãy xương ở {_dich_vung(m.group(2))}"
        if m.group(3):
            cau += f", vị trí {_dich_vi_tri(m.group(3))}"
        return cau + "."
    m = re.fullmatch(r"(.+) of the (.+)", chuan)
    if m and any(k in m.group(1) for k in VI_TRI_ANH_VIET):
        return f"{_dich_vi_tri(m.group(1))} của {_dich_vung(m.group(2))}"

    da_dich = _dich_vung(chuan)
    if da_dich != chuan:
        return da_dich
    return goc
