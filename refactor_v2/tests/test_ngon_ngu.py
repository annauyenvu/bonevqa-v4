import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ngon_ngu import sang_tieng_anh, sang_tieng_viet
from src.utils import is_closed_question

CAU_HOI_VIET = {
    "Ảnh này có gãy xương không?": "Is there a fracture in this X-ray?",
    "Anh nay co gay xuong khong?": "Is there a fracture in this X-ray?",
    "Đây là vùng cơ thể nào?": "Which body region is shown in this X-ray?",
    "vung co the nao": "Which body region is shown in this X-ray?",
    "Có bao nhiêu vị trí gãy?": "How many fracture sites are visible?",
    "Có dụng cụ kim loại/nẹp vít không?": "Is there any orthopedic hardware or implant visible?",
    "Bàn tay có bị gãy không?": "Is the hand fractured?",
    "Ban tay co bi gay khong?": "Is the hand fractured?",
    "Mô tả tổn thương giúp tôi": "Describe the finding.",
    "Vị trí gãy ở đâu?": "Where is the fracture located?",
    "Đây có phải ảnh X-quang bàn tay không?": "Is this an X-ray of the hand?",
    "Có nhiều chỗ gãy không?": "Are there multiple fractures?",
    "Ảnh này chụp bằng kỹ thuật gì?": "What type of imaging is this?",
    "Có bất thường gì không?": "What abnormality is seen in this X-ray?",
}

CAU_HOI_ANH = [
    "Is there a fracture?",
    "Describe the finding.",
    "Which body region is shown in this X-ray?",
    "How many fracture sites are visible?",
    "What abnormality is seen in this X-ray?",
]

DAP_AN = {
    "yes": "có",
    "no": "không",
    "hand": "bàn tay",
    "zero": "không có ổ gãy nào",
    "two": "hai",
    "no abnormality": "không có bất thường",
    "x-ray": "ảnh X-quang",
    "No fracture is seen in the hand.": "Không thấy gãy xương ở bàn tay.",
    "A fracture is seen in the leg at the upper-left.": "Thấy một ổ gãy xương ở chân, vị trí trên bên trái.",
    "Two fractures are seen in the hand and leg at the center.": "Thấy hai ổ gãy xương ở bàn tay và chân, vị trí trung tâm.",
    "upper-left of the hand": "trên bên trái của bàn tay",
    "pneumonia": "pneumonia",
}


def test_cau_hoi_tieng_viet_duoc_anh_xa_dung_mau():
    for cau, ky_vong in CAU_HOI_VIET.items():
        ra, da_dich = sang_tieng_anh(cau)
        assert da_dich, f"chưa nhận ra tiếng Việt: {cau}"
        assert ra == ky_vong, f"{cau} -> {ra} (mong đợi {ky_vong})"


def test_cau_hoi_tieng_anh_duoc_giu_nguyen():
    for cau in CAU_HOI_ANH:
        ra, da_dich = sang_tieng_anh(cau)
        assert not da_dich and ra == cau, f"{cau} bị đổi thành {ra}"


def test_dich_dap_an_sang_tieng_viet():
    for dap_an, ky_vong in DAP_AN.items():
        ra = sang_tieng_viet(dap_an)
        assert ra == ky_vong, f"{dap_an} -> {ra} (mong đợi {ky_vong})"


def test_nhan_dien_cau_hoi_dong():
    assert is_closed_question("Ảnh này có gãy xương không?")
    assert is_closed_question("Is there a fracture?")
    assert not is_closed_question("Describe the finding.")
    assert not is_closed_question("Which body region is shown in this X-ray?")


def main():
    ham = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in ham:
        f()
        print(f"OK {f.__name__}")
    print(f"Đã chạy {len(ham)} test, tất cả đều đạt.")


if __name__ == "__main__":
    main()
