import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import is_choice_question, is_closed_question, is_polar_question

CAU_LUA_CHON = [
    "is the colon more prominent on the patient's right or left side?",
    "is the heart size in this image smaller or larger than normal?",
    "is this supratentorial or infratentorial?",
    "is this an mri or a ct scan?",
    "is the lesion on the left or right side of the brain?",
]

CAU_YES_NO = [
    "Is there a fracture in this X-ray?",
    "Is there any orthopedic hardware or implant visible?",
    "Does the liver show an enhancing mass or lesion?",
    "Are there multiple fractures?",
    "Is this a frontal view X-ray?",
    "Ảnh này có gãy xương không?",
]

CAU_MO = [
    "Which body region is shown in this X-ray?",
    "Describe the finding.",
    "Where is the fracture located?",
    "What abnormality is seen in this X-ray?",
]


def test_cau_lua_chon_khong_bi_dinh_tuyen_thanh_yes_no():
    for q in CAU_LUA_CHON:
        assert is_choice_question(q), f"chưa nhận ra câu lựa chọn: {q}"
        assert not is_closed_question(q), f"câu lựa chọn bị định tuyến thành yes/no: {q}"


def test_cau_yes_no_van_duoc_tra_loi_bang_yes_no():
    for q in CAU_YES_NO:
        assert is_polar_question(q), f"mất nhánh yes/no cho: {q}"


def test_cau_mo_khong_phai_cau_dong():
    for q in CAU_MO:
        assert not is_closed_question(q) and not is_polar_question(q), f"câu mở bị coi là câu đóng: {q}"


def main():
    ham = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in ham:
        f()
        print(f"OK {f.__name__}")
    print(f"Đã chạy {len(ham)} test, tất cả đều đạt.")


if __name__ == "__main__":
    main()
