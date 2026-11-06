"""Audit only TRAIN spatial targets; no model fitting or model-validation labels."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from src.preprocessing import canonicalize_boxes,exif_orientation,canonicalize_raster
from experiments.lnext_spatial_v1 import spatial_targets

ROOT=Path(__file__).resolve().parents[2];P=ROOT/'refactor_v2/reports/step4_localization_v3_1';O=ROOT/'refactor_v2/reports/lnext_spatial_v1_targets2'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2)

def main():
    assert not O.exists();O.mkdir()
    try:
        for path,digest in read(P/'data_seal.json')['artifacts'].items():assert sha(ROOT/path)==digest
        records=read(P/'train.json');validation=read(P/'model_val.json');lookup=read(P/'component_lookup.json')
        canonical=[r for r in records if r['question']=='Is there a fracture in this X-ray?']
        assert len(records)==1164 and len(canonical)==256 and len({r['image_id'] for r in canonical})==256
        assert {r['image_id'] for r in canonical}=={r['image_id'] for r in records}
        assert not {lookup[r['image_id']] for r in canonical}&{lookup[r['image_id']] for r in validation}
        # Analytic geometry fixtures: full letterboxed image, one exact patch,
        # empty image, narrow image and area-preserving placebo/uniform.
        t,w,_=spatial_targets((224,224),[[0,0,16,16]])
        assert t.sum()==1 and t[0]==1 and w.sum()==196
        t,w,_=spatial_targets((100,200),[[0,0,100,200]])
        assert np.allclose(t[w>0],1) and np.allclose((t*w).sum(),98)
        for mode in ('real','placebo','uniform'):
            t,w,_=spatial_targets((111,223),[[10,20,53,99]],mode)
            if mode=='real':reference=float((t*w).sum())
            assert abs(float((t*w).sum())-reference)<1e-5
        t,w,_=spatial_targets((224,224),[]);assert t.sum()==0 and w.sum()==196
        all_targets={mode:{} for mode in ('real','placebo','uniform','off')};images={};positive_mass=0;valid_mass=0;positives=0
        for row in canonical:
            image_id=row['image_id'];path=ROOT/'.evaluation_runtime/data/fracatlas/FracAtlas'/row['image']
            with Image.open(path) as raw:
                orientation=exif_orientation(raw);raw_size=raw.size;canonical_image=canonicalize_raster(raw)
                annotation_size=(row['width'],row['height'])
                if orientation==1:
                    assert annotation_size==raw_size;boxes=row['boxes'];coordinate_rule='identity'
                elif annotation_size==canonical_image.size and annotation_size!=raw_size:
                    # COCO's annotation canvas is already EXIF-canonical. Do not
                    # rotate those boxes a second time.
                    boxes=row['boxes'];coordinate_rule='annotation_canvas_already_canonical'
                elif annotation_size==raw_size and annotation_size!=canonical_image.size:
                    boxes=canonicalize_boxes(row['boxes'],raw_size,orientation);coordinate_rule='raw_canvas_transformed'
                elif not row['boxes']:
                    boxes=None;coordinate_rule='empty_no_coordinate_effect'
                else:
                    raise AssertionError('Ambiguous nonidentity EXIF annotation canvas: '+image_id)
                assert bool(boxes)==bool(row['fractured'])==(row['answer']=='yes'),image_id
                size=canonical_image.size
            images[image_id]={'path':str(path),'sha256':sha(path),'raw_size':raw_size,'canonical_size':size,'annotation_size':annotation_size,'orientation':orientation,'coordinate_rule':coordinate_rule,'box_count':len(boxes or [])}
            real_mass=None
            for mode in all_targets:
                t,w,meta=spatial_targets(size,boxes,mode)
                all_targets[mode][image_id]={'target':t.tolist(),'weight':w.tolist(),'geometry':meta}
                mass=float((t*w).sum())
                if mode=='real':
                    real_mass=mass;positive_mass+=meta['positive_pixels'];valid_mass+=meta['image_pixels']
                elif mode!='off':assert abs(mass-real_mass)<1e-4,image_id
            a=np.array(all_targets['real'][image_id]['target']);b=np.array(all_targets['placebo'][image_id]['target']);weight=np.array(all_targets['real'][image_id]['weight'])
            images[image_id]['coarse_placebo_overlap_mass']=float((np.minimum(a,b)*weight).sum())
            if row['answer']=='yes':
                positives+=1;assert real_mass>0,image_id
        assert positives==46
        for mode,data in all_targets.items():write(O/f'targets_{mode}.json',data)
        write(O/'target_audit.json',{'passed':True,'images':images,'TRAIN_images':256,'TRAIN_positive_images':positives,'TRAIN_QA':len(records),'modelval_component_disjoint':True,'foreground_pixels':positive_mass,'valid_image_pixels':valid_mass,'spatial_positive_weight':(valid_mass-positive_mass)/positive_mass,'geometry_fixtures_passed':True,'model_gates_pending':True,'quality_released':False,'dependencies':{str(P/n):sha(P/n) for n in ('train.json','model_val.json','data_seal.json','component_lookup.json')}})
        write(O/'target_seal.json',{'artifacts':{str(p):sha(p) for p in O.glob('*.json')},'code':{str(Path(__file__)):sha(Path(__file__)),str(ROOT/'refactor_v2/experiments/lnext_spatial_v1.py'):sha(ROOT/'refactor_v2/experiments/lnext_spatial_v1.py')},'quality_released':False})
        print(json.dumps({'target_audit_passed':True,'images':256,'positive_images':positives,'spatial_positive_weight':(valid_mass-positive_mass)/positive_mass,'model_gates_pending':True}))
    except Exception as e:
        write(O/'preparation_failure.json',{'error':repr(e),'preserve_all_artifacts':True});raise

if __name__=='__main__':main()
