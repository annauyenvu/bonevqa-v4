"""Five zero-QA initial checkpoints; fresh-process TRAIN inference roundtrip only."""
import argparse
import copy
import gc
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
import torch
from src.data import build_dataset, collate_fn
from src.train import build_split, load_init_weights
from src.models.bonevqa import BoneVQAModel
from src.protocol import records_of
from src.utils import set_seed
from experiments.run_vl_bounded_pilot_v3_1 import check_environment
from experiments.run_T_bounded_current_v3_1 import hashes
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree
from experiments.audit_fresh_exact_upstream_fracatlas_v3_1 import ROOT, sha, dump

OUT=ROOT/'refactor_v2/reports/perfold_PREtask_roundtrip_v3_1'
UPSTREAM=ROOT/'refactor_v2/checkpoints/revised_roco_alignment_v3/best.pt'
DRAFT=ROOT/'refactor_v2/reports/NESTED_DESIGN_REQUIREMENTS_V3_1.json'
SEAL=ROOT/'refactor_v2/reports/STEP1_DATA_SEAL_V3_ALIGNMENT.json'
CFG=ROOT/'refactor_v2/configs/revised_sealed_candidates_v3/b.json'


def inputs(fold):
    design=json.loads(DRAFT.read_text());base=ROOT/f'.evaluation_runtime/data/fracatlas_component_cv_v3/fold_{fold}'
    sealed=json.loads(SEAL.read_text());train=base/'vqa_train.json';vocab=base/'answer_vocab_runtime_normalized_v3.json'
    for p in (train,vocab):assert sha(p)==sealed['artifacts'][str(p.relative_to(ROOT))]
    detpath=base/'detector_split_v3.json';assert sha(detpath)==design['folds'][str(fold)]['detector_metadata_sha256']
    det=json.loads(detpath.read_text());assert det['source_vqa']['train']=={'path':str(train),'sha256':sha(train)}
    cfg=json.loads(CFG.read_text());loader=copy.deepcopy(cfg['data']['loader_kwargs'])
    loader.update(seal_path=str(SEAL),project_root=str(ROOT),root=str(ROOT/'.evaluation_runtime/data/fracatlas/FracAtlas'),
        with_masks=False,with_boxes=False,boxes_from=None,boxes_as_masks=False,
        sealed_records_by_split={'train':str(train),'model_val':None,'calibration':None,'test':None})
    ds=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,loader);rows=records_of(ds)
    assert len(rows)==design['folds'][str(fold)]['train']['QA']
    selected=[];used=set();groups=det['component_by_image']
    for kind,answer in [('CLOSED','yes'),('CLOSED','no'),('OPEN',None),('OPEN',None)]:
        i=next(i for i,r in enumerate(rows) if r['answer_type']==kind and groups[r['image_id']] not in used
            and (answer is None or (r['question']=='Is there a fracture in this X-ray?' and r['answer']==answer)))
        selected.append(i);used.add(groups[rows[i]['image_id']])
    assert len(used)==4
    counts=Counter(r['answer'] for r in rows if r['question']=='Is there a fracture in this X-ray?')
    modelcfg=copy.deepcopy(cfg['model']);modelcfg.update(module_init_seed=1771,fracture_positive_weight=counts['no']/counts['yes'])
    assert not any(modelcfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt'))
    batch=collate_fn([ds[i] for i in selected]);public={'images':batch['images'],'questions':batch['questions']}
    meta={'TRAIN_sha256':sha(train),'vocabulary_sha256':sha(vocab),'vocabulary_size':len(json.loads(vocab.read_text())),
        'TRAIN_canonical_counts':dict(counts),'TRAIN_positive_weight':counts['no']/counts['yes'],
        'TRAIN_indices':selected,'TRAIN_image_ids':[rows[i]['image_id'] for i in selected],
        'TRAIN_components':[groups[rows[i]['image_id']] for i in selected],
        'image_sha256':[sha(ds.root/rows[i]['image']) for i in selected]}
    return cfg,modelcfg,json.loads(vocab.read_text()),public,meta


def predict(model,public):
    model.eval();before=hashes(model.trainable_state_dict())
    with torch.no_grad():records=model.predict_batch(public)
    assert hashes(model.trainable_state_dict())==before
    return [{k:v for k,v in r.items() if k!='masks'} for r in records]


def validate():
    check_environment();p=json.loads((OUT/'protocol.json').read_text())
    for name,h in p['dependency_sha256'].items():assert sha(Path(name))==h
    assert sha(UPSTREAM)==p['upstream_sha256']
    return p


def replay():
    validate();assert not (OUT/'fresh_process_replay.json').exists()
    first=json.loads((OUT/'exports.json').read_text());verified={}
    for fold in range(5):
        key=str(fold);path=OUT/f'fold_{fold}_PREtask.pt';expected=first[key]
        assert sha(path)==expected['checkpoint_sha256']
        cfg,modelcfg,vocab,public,meta=inputs(fold);assert meta==expected['TRAIN_binding']
        payload=torch.load(path,map_location='cpu',weights_only=False)
        assert payload['answer_vocab']==vocab and payload['extra']['quality_optimizer_updates']==0
        assert len(payload['state'])==263 and finite_tree(payload['state'])
        set_seed(90211);model=BoneVQAModel(payload['config'],vocab,device='cpu')
        model.load_trainable(str(path))
        assert model.cfg==payload['config'] and hashes(model.trainable_state_dict())==expected['all263_tensor_sha256']
        records=predict(model,public);assert records==expected['public_TRAIN_predictions']
        verified[key]={'passed':True,'all263_tensors_and_all4_public_predictions_exact':True,
            'checkpoint_sha256':sha(path),'vocabulary_size':len(vocab)}
        del model,payload;gc.collect()
        dump(OUT/'replay_progress.json',{'completed_folds':list(verified),'pid':os.getpid()})
    dump(OUT/'fresh_process_replay.json',{'passed':True,'constructor_seed':90211,'folds':verified,
        'optimizer_updates':0,'reserved_role_inference':False,'long_training_authorized':False})
    print('Fresh-process five-fold zero-QA checkpoint roundtrip exact',flush=True)


def run():
    check_environment();assert not OUT.exists()
    assert sha(DRAFT)=='0a225a950cc4d42339075c652be7a9bd07d4a4fcdbb15b29bf389b04ea56565c'
    assert sha(UPSTREAM)=='08cd132ef1097da8f64758225b3f66b7a5329a5d008accfe84e352e94cb98302'
    paths=[Path(__file__).resolve(),DRAFT,SEAL,CFG,
        ROOT/'refactor_v2/reports/STEP3_TECHNICAL_REFERENCE_SEAL_V3.json',
        ROOT/'refactor_v2/reports/fullfold0_uniform_lengths_v3_1/report.json',
        ROOT/'refactor_v2/reports/fullfold0_uniform_lengths_v3_1/independent_artifact_verification.json']
    for fold in range(5):
        base=ROOT/f'.evaluation_runtime/data/fracatlas_component_cv_v3/fold_{fold}'
        paths.extend(base/n for n in ('vqa_train.json','answer_vocab_runtime_normalized_v3.json','detector_split_v3.json'))
    OUT.mkdir();dump(OUT/'protocol.json',{'scope':'Zero-QA-update PREtask exports and TRAIN-only fresh-process roundtrip, not a training release',
        'dependency_sha256':{str(p):sha(p) for p in paths},'upstream_sha256':sha(UPSTREAM),
        'folds':list(range(5)),'initialization_seed':42,'module_init_seed':1771,'fresh_reload_constructor_seed':90211,
        'optimizer_updates':0,'quality_training_allowed':False,'modelval_calibration_outer_inference_allowed':False,
        'within_fold_B_uniform_same_initial_file':True,'long_training_authorized':False})
    started=time.perf_counter();results={};shared=None
    dump(OUT/'progress.json',{'status':'exporting_zeroQA_initials','pid':os.getpid(),'completed_folds':[]})
    try:
        for fold in range(5):
            cfg,modelcfg,vocab,public,meta=inputs(fold);set_seed(42)
            model=BoneVQAModel(modelcfg,vocab,device='cpu')
            heads={k:v.clone() for k,v in model.trainable_state_dict().items() if k.startswith(('closed_head.','fracture_head.'))}
            load_init_weights(model,str(UPSTREAM),cfg['checkpoint_allow_mismatch_prefixes'],False)
            model.load_state_dict(heads,strict=False)
            initial=hashes(model.trainable_state_dict());assert len(initial)==263 and finite_tree(model.trainable_state_dict())
            common={k:v for k,v in initial.items() if not k.startswith(('closed_head.','fracture_head.'))}
            if shared is None:shared=common
            else:assert common==shared
            if fold==0:
                old=json.loads((ROOT/'refactor_v2/reports/fullfold0_uniform_lengths_v3_1/report.json').read_text())
                assert initial==old['all263_initial_tensor_hashes']
            path=OUT/f'fold_{fold}_PREtask.pt'
            model.save(str(path),{'scope':'technical_PREtask_zeroQA_initialization_ONLY','fold':fold,
                'quality_optimizer_updates':0,'upstream_sha256':sha(UPSTREAM),'TRAIN_binding':meta,
                'protocol_sha256':sha(OUT/'protocol.json'),'scientific_execution_released':False})
            records=predict(model,public)
            results[str(fold)]={'checkpoint_sha256':sha(path),'all263_tensor_sha256':initial,
                'TRAIN_binding':meta,'public_TRAIN_predictions':records}
            dump(OUT/'exports.json',results)
            dump(OUT/'progress.json',{'status':'exporting_zeroQA_initials','pid':os.getpid(),'completed_folds':list(results)})
            del model,heads;gc.collect()
        dump(OUT/'progress.json',{'status':'fresh_process_reload_running','pid':os.getpid(),'completed_folds':list(results)})
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--replay'],env=os.environ.copy(),check=True)
        replayed=json.loads((OUT/'fresh_process_replay.json').read_text());assert replayed['passed']
        validate()
        report={'passed':True,'protocol_sha256':sha(OUT/'protocol.json'),'exports_sha256':sha(OUT/'exports.json'),
            'fresh_process_replay_sha256':sha(OUT/'fresh_process_replay.json'),
            'five_fold_initializations_and_fresh_process_roundtrips_exact':True,
            'shared_non_task_head_tensor_count':len(shared),'shared_non_task_head_tensors_exact_across_folds':True,
            'vocabulary_sizes':{f:r['TRAIN_binding']['vocabulary_size'] for f,r in results.items()},
            'elapsed_seconds_including_reload':time.perf_counter()-started,
            'saved_checkpoints':5,'saved_weight_scope':'PREtask only, ZERO QA optimizer updates',
            'within_fold_B_uniform_initial_parity':'Same exported checkpoint file required for both arms; not a trained control experiment',
            'optimizer_updates':0,'modelval_calibration_outer_inference_used':False,
            'long_training_authorized':False,'step3_complete':False,'Final_Solution_selected':False,'production_released':False}
        dump(OUT/'report.json',report);dump(OUT/'progress.json',{'status':'complete_pending_artifact_audit','report_sha256':sha(OUT/'report.json')})
        print(json.dumps(report,indent=2),flush=True)
    except Exception as error:
        dump(OUT/'progress.json',{'status':'failed_preserved_no_retry','error':repr(error),'completed_folds':list(results)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    replay() if args.replay else run()
