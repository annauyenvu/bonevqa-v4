"""Classic Vietnamese UI with the real local-noaux research checkpoint."""
import os,threading
from pathlib import Path
import numpy as np
import torch
from fastapi.responses import FileResponse
from experiments import serve_local_noaux_demo_v1 as runtime
from app.inference import build_visual_prompts
from src.data.fracatlas_vqa import FracAtlasVQADataset

ROOT=runtime.ROOT
os.environ['BONEVQA_FEEDBACK']=str(ROOT/'refactor_v2/reports/local_noaux_demo_feedback_v2.jsonl')
class Engine:
    name='Local-noaux · fold 0 LAST · chưa promotion'
    model_loaded=True
    device='cpu (2 threads)'
    calibration_provenance=None
    def __init__(self):
        self.model=runtime.load('local_real_noaux');self.detector=runtime.detector
        self.detector_provenance={'checkpoint':'fold0 detector640 LAST','score_threshold':.05,'max_boxes':3,'mask_type':'rectangular proposals, NOT SAM segmentation'}
    def detect(self,image):return self.detector.predict(image)
    def segment(self,image,boxes=None,points=None):
        if points:raise ValueError('Bản local-token này không dùng SAM point prompts')
        with runtime.lock,torch.no_grad():
            boxes=self.detect(image) if boxes is None else boxes
            masks=FracAtlasVQADataset._boxes_to_masks(boxes,image.size) if boxes else []
            return np.asarray(masks,dtype=np.uint8) if len(masks) else np.zeros((0,image.height,image.width),dtype=np.uint8)
    def answer(self,image,question,boxes=None,prompt_kind='contour',masks=None):
        with runtime.lock,torch.no_grad():
            boxes=self.detect(image) if boxes is None else boxes
            masks=FracAtlasVQADataset._boxes_to_masks(boxes,image.size) if boxes else []
            out=self.model.predict_batch({'images':[image],'questions':[question],'boxes':[boxes or None],'masks':[masks]})[0]
            masks=np.asarray(masks,dtype=np.uint8) if len(masks) else np.zeros((0,image.height,image.width),dtype=np.uint8)
            return dict(answer=out['answer'],answer_type=out.get('answer_type','CLOSED'),confidence=float(out.get('confidence',0)),masks=masks,prompt_image=build_visual_prompts(image,masks,prompt_kind),lens_image=image.copy(),detector_boxes=boxes)

import app.inference as inference
inference._ENGINE=Engine()
from app.api import app
# Preserve all original sealed source files; override only this demo's root route.
app.router.routes[:]=[r for r in app.router.routes if getattr(r,'path',None)!='/']
@app.get('/')
def index():return FileResponse(ROOT/'refactor_v2/experiments/local_noaux_classic_demo_v2.html')
