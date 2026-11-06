"""Đo khởi động, RAM, độ trễ và kiểm thử đầu cuối của API demo V4 trên một tiến trình riêng ở cổng 8021."""
import io, json, os, statistics as st, subprocess, sys, time
from pathlib import Path
import requests
from PIL import Image
ROOT = Path('/Users/uyen/Downloads/BONEVQA'); R = ROOT / 'refactor_v2'; W = R / 'experiments/v4_thesis_demo'; OUT = R / 'reports/thesis_v4_checks_v1/system_benchmark.json'; FB = R / 'reports/v4_demo_feedback.jsonl'; U = 'http://127.0.0.1:8021'
env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=str(ROOT / '.evaluation_runtime/hf_cache'), BONEVQA_DATA=str(ROOT / '.evaluation_runtime/data'), PYTHONPATH=f'{R}:{ROOT}/.evaluation_runtime/deps')
t0 = time.time(); srv = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'experiments.serve_v4_thesis_demo:app', '--host', '127.0.0.1', '--port', '8021', '--log-level', 'warning'], cwd=R, env=env)
try:
    for _ in range(600):
        try:
            if requests.get(U + '/api/health', timeout=1).ok: break
        except Exception: time.sleep(.5)
    startup = time.time() - t0
    ex = sorted((W / 'examples').glob('*.jpg')); CLOSED = 'Is there a fracture in this X-ray?'; OPEN = ['Which body region is shown in this X-ray?', 'Describe the finding.']; REP = 5
    def post(path, f, **data): return requests.post(U + path, files={'image': (f.name, f.read_bytes(), 'image/jpeg')}, data=data, timeout=300)
    for f in ex[:2]: post('/api/ask', f, question=CLOSED)
    lat = dict(detect=[], closed=[], open=[]); rec = []
    for f in ex:
        for _ in range(REP):
            lat['detect'].append(post('/api/detect', f).json()['latency_ms'])
            r = post('/api/ask', f, question=CLOSED).json(); lat['closed'].append(r['latency_ms']); rec.append(r['p_yes'])
            for q in OPEN: lat['open'].append(post('/api/ask', f, question=q).json()['latency_ms'])
    rss = float(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(srv.pid)])) / 1024 / 1024
    tests = []
    def t(name, want, ok): tests.append(dict(name=name, want=want, ok=bool(ok))); print(name, ok, flush=True)
    r = requests.get(U + '/api/health'); t('GET /api/health', 'Trạng thái ok, tên mô hình', r.ok and r.json()['status'] == 'ok')
    r = requests.get(U + '/api/info'); t('GET /api/info', 'Có kết quả và giới hạn', r.ok and r.json().get('gioi_han'))
    r = requests.get(U + '/api/examples'); t('GET /api/examples', 'Sáu ảnh mẫu', r.ok and len(r.json()['examples']) == 6)
    r = post('/api/detect', ex[3]); t('POST /api/detect', 'Tối đa ba vùng có điểm', r.ok and len(r.json()['regions']) <= 3 and r.json()['regions'])
    a = post('/api/ask', ex[3], question=CLOSED).json(); b = post('/api/ask', ex[3], question=CLOSED).json(); t('Hỏi lặp cùng ảnh, câu', 'p_yes giống nhau', a['p_yes'] == b['p_yes'] and a['answer_type'] == 'CLOSED')
    r = post('/api/ask', ex[3], question='Ảnh này có gãy xương không?').json(); t('Câu tiếng Việt theo mẫu', 'Ánh xạ về câu canonical', r['question_en'] == CLOSED and r['is_fracture_question'])
    r = post('/api/ask', ex[3], question='Please write a poem about bones').json(); t('Câu ngoài tập mẫu', 'Cờ evaluated_question = false', r['evaluated_question'] is False)
    b_ = io.BytesIO(); Image.open(ex[0]).convert('RGB').save(b_, 'PNG'); r = requests.post(U + '/api/ask', files={'image': ('a.png', b_.getvalue(), 'image/png')}, data=dict(question=CLOSED)); t('Ảnh PNG', 'Trả lời 200', r.ok)
    r = requests.post(U + '/api/ask', files={'image': ('a.txt', b'not an image', 'text/plain')}, data=dict(question=CLOSED)); t('Tệp không phải ảnh', 'Lỗi 400', r.status_code == 400)
    r = requests.post(U + '/api/ask', files={'image': ('a.jpg', b'0' * (20 * 1024 * 1024 + 5), 'image/jpeg')}, data=dict(question=CLOSED)); t('Ảnh lớn hơn 20 MB', 'Lỗi 413', r.status_code == 413)
    r = post('/api/ask', ex[3], question=' '); t('Câu hỏi rỗng', 'Lỗi 400 hoặc 422', r.status_code in (400, 422))
    r = post('/api/ask', ex[3], question=CLOSED, source='xyz'); t('Nguồn gợi ý sai', 'Lỗi 400', r.status_code == 400)
    r = post('/api/ask', ex[3], question=CLOSED, source='manual', boxes='[[100,100,500,500]]'); t('Vùng vẽ hợp lệ', 'Dùng đúng một vùng', r.ok and len(r.json()['regions']) == 1 and r.json()['regions'][0]['score'] is None)
    r = post('/api/ask', ex[3], question=CLOSED, source='manual', boxes=json.dumps([[0, 0, 9, 9]] * 4)); t('Bốn vùng vẽ', 'Lỗi 400', r.status_code == 400)
    n0 = FB.read_text().splitlines() if FB.exists() else []
    r = requests.post(U + '/api/feedback', data=dict(question='q', answer='a', verdict='dung')); n1 = FB.read_text().splitlines(); t('Phản hồi hợp lệ', 'Thêm một dòng JSONL', r.ok and len(n1) == len(n0) + 1)
    FB.write_text(''.join(l + '\n' for l in n0))
    r = requests.post(U + '/api/feedback', data=dict(question='q', answer='a', verdict='xx')); t('Phản hồi sai giá trị', 'Lỗi 400', r.status_code == 400)
    def summ(v): v = sorted(v); return dict(n=len(v), p50=st.median(v), p95=v[int(.95 * (len(v) - 1))], max=v[-1], min=v[0])
    F0 = R / 'reports/local_noaux_refinement_v1/fold_0'; rc = json.loads((F0 / 'detector/receipt.json').read_text())
    OUT.write_text(json.dumps(dict(startup_s=startup, rss_gb=rss, repeats=REP, threads=4, machine='Apple M4 Pro, 24 GB', latency_ms={k: summ(v) for k, v in lat.items()}, tests=tests, ckpt_mb=(F0 / 'local_real_noaux/LAST.pt').stat().st_size / 1e6, det_mb=Path(rc['checkpoint']).stat().st_size / 1e6, sensitive_threshold=requests.get(U + '/api/health').json()['sensitive_threshold']), ensure_ascii=False, indent=1))
    print(OUT.read_text()[:900])
finally:
    srv.terminate()
