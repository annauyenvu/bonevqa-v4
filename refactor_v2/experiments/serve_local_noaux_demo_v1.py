"""Local-only research demo; isolated from immutable experiment sources."""
import gc,hashlib,io,json,threading,time
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from fastapi.responses import FileResponse
from PIL import Image,ImageDraw
from experiments.vnext_model_v1 import VNextModel
from experiments.detector640_checkpoint_v3 import load_bound_640
from src.detector_runtime import FractureDetectorRuntime
from src.preprocessing import canonicalize_raster
from src.data.fracatlas_vqa import FracAtlasVQADataset
from app.ngon_ngu import sang_tieng_anh,sang_tieng_viet
import base64

ROOT=Path('/Users/uyen/Downloads/BONEVQA');O=ROOT/'refactor_v2/reports/local_noaux_refinement_v1'
torch.set_num_threads(2)
app=FastAPI(title='BoneVQA research demo')
lock=threading.Lock();model=None;loaded=None;detector=None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(arm):
    global model,loaded,detector
    assert arm in ('B','local_real_noaux')
    if loaded!=arm:
        model=None;gc.collect()
        folder=O/'fold_0'/arm;p=folder/'LAST.pt'
        assert sha(p)==json.loads((folder/'completed.json').read_text())['checkpoint_sha256']
        model=VNextModel.load(str(p),device='cpu').eval();loaded=arm
    if arm!='B' and detector is None:
        receipt=json.loads((O/'fold_0/detector/receipt.json').read_text());assert sha(receipt['checkpoint'])==receipt['sha256']
        detector=object.__new__(FractureDetectorRuntime);detector.model,_=load_bound_640(receipt['checkpoint'])
        detector.device=torch.device('cpu');detector.image_size=640;detector.score_threshold=.05;detector.max_boxes=3
    return model

@app.get('/')
def index():return FileResponse(ROOT/'refactor_v2/experiments/local_noaux_demo_v1.html')
@app.get('/health')
def health():return {'status':'ok','mock':False,'loaded':loaded,'fold':0,'head':'original LAST, threshold 0.5','CPU_threads':2}
@app.post('/ask')
def ask(image:UploadFile=File(...),question:str=Form(...),arm:str=Form('local_real_noaux')):
    if arm not in ('B','local_real_noaux'):raise HTTPException(400,'Unknown model')
    data=image.file.read(20*1024*1024+1)
    if len(data)>20*1024*1024:raise HTTPException(413,'Ảnh vượt 20 MB')
    try:im=Image.open(io.BytesIO(data));im.load();im=canonicalize_raster(im)
    except Exception:raise HTTPException(400,'Không đọc được ảnh')
    q=question.strip()
    if not q or len(q)>500:raise HTTPException(400,'Câu hỏi cần dài 1–500 ký tự')
    start=time.time()
    with lock,torch.no_grad():
        m=load(arm);en,translated=sang_tieng_anh(q)
        boxes=detector.predict(im) if arm!='B' else []
        masks=FracAtlasVQADataset._boxes_to_masks(boxes,im.size) if boxes else []
        batch={'images':[im],'questions':[en]}
        if arm!='B':batch.update(boxes=[boxes or None],masks=[masks])
        out=m.predict_batch(batch)[0]
        preview=im.copy();draw=ImageDraw.Draw(preview)
        for box in boxes:draw.rectangle(box,outline='orange',width=max(2,im.width//400))
        preview.thumbnail((1000,1000));buf=io.BytesIO();preview.save(buf,format='PNG')
        return {'answer':sang_tieng_viet(out['answer']) if translated else out['answer'],'answer_en':out['answer'],'question_en':en,'p_yes':out.get('p_yes'),'confidence':out.get('confidence'),'arm':arm,'fold':0,'boxes':boxes,'seconds':round(time.time()-start,2),'preview':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),'warning':'Nghiên cứu; không dùng chẩn đoán. Khung là proposal detector, không phải xác nhận vị trí gãy. Demo batch1 có thể lệch số học rất nhỏ so batch4 đánh giá.'}
