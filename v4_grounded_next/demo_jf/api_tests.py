"""End-to-end tests of the running JF demo API (:8021). python3 -I v4_grounded_next/demo_jf/api_tests.py -> api_tests.json"""
import json, subprocess, os, tempfile
from pathlib import Path
H = Path(__file__).resolve().parent; EX = H / "web/examples"
def curl(*a):
    r = subprocess.run(["curl", "-s", "-w", "\n%{http_code}", *a], capture_output=True, text=True).stdout.rsplit("\n", 1)
    try: body = json.loads(r[0])
    except Exception: body = r[0][:80]
    return int(r[1]), body
U = "localhost:8021"
tests = []
def t(name, expect, ok, got): tests.append({"test": name, "expect": expect, "pass": bool(ok), "got": got})
c, b = curl(f"{U}/api/health"); t("GET /api/health", "200, mô hình JF", c == 200 and b["model"].startswith("JF"), b.get("model"))
c, b = curl("-F", f"image=@{EX/'mau_4.jpg'}", f"{U}/api/detect"); t("POST /api/detect, ảnh gãy", "200, có hộp lớp gãy", c == 200 and any(r["cls"] == 0 for r in b["regions"]), len(b["regions"]))
c, b = curl("-F", f"image=@{EX/'u_BT_IMG000002.jpg'}", f"{U}/api/detect"); t("POST /api/detect, ảnh u", "200, có hộp lớp tổn thương", c == 200 and any(r["cls"] == 1 for r in b["regions"]), len(b["regions"]))
c, b = curl("-F", f"image=@{EX/'mau_4.jpg'}", "-F", "question=Ảnh này có gãy xương không?", f"{U}/api/ask"); t("Câu gãy tiếng Việt", "định tuyến câu chuẩn, có quyết định tin cậy", c == 200 and b["question_en"] == "Is there a fracture in this X-ray?" and b["trust"]["decision"] in ("answer", "defer"), b.get("trust", {}).get("decision"))
c, b = curl("-F", f"image=@{EX/'u_BT_IMG000002.jpg'}", "-F", "question=Có u xương không?", f"{U}/api/ask"); t("Câu có u tiếng Việt", "hộp lớp tổn thương, có quyết định tin cậy", c == 200 and b["routed_class"] == "tổn thương xương" and b["trust"]["decision"] in ("answer", "defer"), b.get("trust", {}).get("decision"))
c, b = curl("-F", f"image=@{EX/'u_BT_IMG000002.jpg'}", "-F", "question=Tổn thương lành hay ác tính?", f"{U}/api/ask"); t("Câu lành/ác", "câu chuẩn ác tính, có quyết định tin cậy", c == 200 and b["question_en"] == "Is the bone lesion malignant?", b.get("trust", {}).get("decision"))
c, b = curl("-F", f"image=@{EX/'u_BT_IMG000002.jpg'}", "-F", "question=What is the most likely diagnosis?", f"{U}/api/ask"); t("Câu chẩn đoán", "ba chẩn đoán đứng đầu", c == 200 and len(b["trust"]["top3"]) == 3, [x["answer_vi"] for x in b["trust"]["top3"]])
c, b = curl("-F", f"image=@{EX/'mau_1.jpg'}", "-F", "question=Which body region is shown in this X-ray?", f"{U}/api/ask"); t("Câu mở khác", "trả lời, đánh dấu chưa hiệu chỉnh", c == 200 and b["trust"]["decision"] == "uncalibrated", b.get("answer"))
c, b = curl("-F", f"image=@{EX/'mau_1.jpg'}", "-F", "question=", f"{U}/api/ask"); t("Câu hỏi rỗng", "400 hoặc 422", c in (400, 422), c)
with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
    f.write(os.urandom(21 * 1024 * 1024)); f.flush()
    c, b = curl("-F", f"image=@{f.name}", "-F", "question=Có u xương không?", f"{U}/api/ask"); t("Ảnh lớn hơn 20 MB", "413", c == 413, c)
with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
    f.write(b"not an image"); f.flush()
    c, b = curl("-F", f"image=@{f.name}", "-F", "question=Có u xương không?", f"{U}/api/ask"); t("Tệp không phải ảnh", "400", c == 400, c)
c, b = curl("-F", "question=q", "-F", "answer=a", "-F", "verdict=xyz", f"{U}/api/feedback"); t("Phản hồi sai định dạng", "400", c == 400, c)
c, b = curl("-F", "question=api test", "-F", "answer=yes", "-F", "verdict=dung", f"{U}/api/feedback"); t("Ghi phản hồi", "200, đã lưu", c == 200 and b.get("saved"), b)
(H / "api_tests.json").write_text(json.dumps(tests, ensure_ascii=False, indent=1))
for x in tests: print(("PASS" if x["pass"] else "FAIL"), x["test"], "|", x["got"])
