"""Join existing public predictions and pixel-coordinate proposal/GT audit only."""
import json
import hashlib
from PIL import Image, ImageOps
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'reports'
P = R / 'step4_localization_v3_1'
O = R / 'step7_synthesis_v3_1/localization_saved.json'
sources = {}
def read(p):
    raw = p.read_bytes(); sources[str(p)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)
def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0
def main():
    assert not O.exists()
    q='Is there a fracture in this X-ray?'
    meta={r['image_id']:r for r in read(P/'model_val.json') if r['question']==q}
    boxes=read(P/'boxes.json'); placebo=read(P/'placebo.json');prov=read(P/'proposal_provenance.json')
    predictions={a:{r['image_id']:r for r in read(P/f'run_{a}/epoch_4_evaluation.json')['records'] if r['question']==q} for a in ('B','V_real','V_placebo')}
    assert len(meta)==321
    rows=[]; orientation_checks=[]
    for image,m in meta.items():
        assert all(set(p)==set(meta) for p in predictions.values())
        gt=m['boxes'] or []; real=boxes[image]; fake=placebo[image]
        width,height=prov['images'][image]['width'],prov['images'][image]['height']
        path=R.parents[1]/'.evaluation_runtime/data/fracatlas/FracAtlas'/m['image']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==prov['images'][image]['sha256']
        with Image.open(path) as im:
            orientation=int(im.getexif().get(274,1) or 1)
            assert ImageOps.exif_transpose(im).size==(width,height)
            if orientation!=1:
                assert not gt, 'Oriented GT requires explicit coordinate transform review'
                orientation_checks.append({'image_id':image,'EXIF':orientation,'raw':list(im.size),'canonical':[width,height],'GT_boxes':0})
        for b in gt+real+fake:
            assert 0<=b[0]<b[2]<=width and 0<=b[1]<b[3]<=height
        hit={a:[max((iou(g,b) for b in bs),default=0) for g in gt] for a,bs in [('real',real),('placebo',fake)]}
        rows.append({'image_id':image,'positive':m['answer']=='yes','GT_boxes':len(gt),'proposal_count':len(real),'best_IoU':max(hit['real'],default=None),'GT_box_IoUs':hit,'positive_predictions':{a:p[image]['p_yes']>=.5 for a,p in predictions.items()}})
    pos=[r for r in rows if r['positive']];fn=[r for r in pos if not r['positive_predictions']['B']]
    def summary(rs):
        return {'images':len(rs),'proposal_present':sum(r['proposal_count']>0 for r in rs),'GT_available':sum(r['GT_boxes']>0 for r in rs),'hit025':sum(r['best_IoU'] is not None and r['best_IoU']>=.25 for r in rs),'hit05':sum(r['best_IoU'] is not None and r['best_IoU']>=.5 for r in rs),'predicted_positive':{a:sum(r['positive_predictions'][a] for r in rs) for a in predictions}}
    result={'scope':'Descriptive inspected modelval; canonical pixel-coordinate boxes; GT audit only; no fitting or inference','sources':sources,'orientation_checks':orientation_checks,'positive':summary(pos),'baseline_FN':summary(fn),'baseline_TP':summary([r for r in pos if r['positive_predictions']['B']]),'negative':summary([r for r in rows if not r['positive']]),'rows':rows}
    O.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('sources','rows')}))
if __name__=='__main__':main()
