"""Post-hoc TRAIN-only FN/TN mechanism probe. No updates or efficacy claims."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
FIT=ROOT/'refactor_v2/reports/fullfold0_TRAIN_canonical_diagnostic_v3_1'
PARENT=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
OUT=ROOT/'refactor_v2/reports/error_bearing_TRAIN_probe_v3_1'
MODES=('token_mean','sequence_uniform')
QUESTION='Is there a fracture in this X-ray?'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare():
    assert not OUT.exists()
    parent=read(PARENT/'protocol.json');cfgpath=parent['configs'][MODES[0]]['path'];cfg=read(cfgpath)
    train=Path(cfg['data']['loader_kwargs']['sealed_records_by_split']['train']);rows=read(train)
    assert read(FIT/'independent_artifact_audit.json')['report_sha256']==sha(FIT/'report.json')
    preds={r['image_id']:r for r in read(FIT/'token_mean.json')['records']}
    b=cfg['train']['component_batching'];lookup={image:g for g,images in enumerate(read(b['manifest'])['components']) for image in images}
    chosen=[];used=set()
    for label in ('yes','no','yes','no'):
        i=next(i for i,r in enumerate(rows) if r['question']==QUESTION and r['answer']==label
            and preds[r['image_id']]['p_yes']<.5 and lookup[r['image_id']] not in used)
        chosen.append(i);used.add(lookup[rows[i]['image_id']])
    deps={str(Path(__file__).resolve()):sha(__file__),str(train):sha(train)}
    deps.update(read(PARENT/'runtime_binding.json')['dependency_sha256'])
    hybrid=ROOT/'refactor_v2/reports/hybrid_optimizer_restart_gate_v3_1'
    assert read(hybrid/'independent_artifact_audit.json')['passed']
    for path in [FIT/'report.json',FIT/'token_mean.json',FIT/'independent_artifact_audit.json',hybrid/'report.json',hybrid/'independent_artifact_audit.json',Path(b['manifest']),Path(b['seal'])]:deps[str(path)]=sha(path)
    targets={}
    for mode in MODES:
        c=parent['configs'][mode];config=read(c['path']);assert sha(c['path'])==c['sha256']
        path=ROOT/config['checkpoint_dir']/config['run_name']/'last.pt'
        digest=read(PARENT/(mode+'_completion.json'))['last_sha256'];assert sha(path)==digest
        targets[mode]={'path':str(path),'sha256':digest}
    OUT.mkdir()
    dump(OUT/'protocol.json',{'scope':'Post-hoc native-error-selected TRAIN mechanism diagnosis, NOT quality or unbiased arm comparison',
        'dependencies_sha256':deps,'config':cfgpath,'TRAIN_manifest':str(train),'targets':targets,
        'indices':chosen,'image_ids':[rows[i]['image_id'] for i in chosen],
        'selection':'First native TRAIN FN/ TN/ FN/ TN on unused known components in original TRAIN order; no modelval-based selection',
        'native_selection_probabilities':[preds[rows[i]['image_id']]['p_yes'] for i in chosen],
        'examples':4,'components':4,'positive_FN_examples':2,'negative_TN_controls':2,
        'public_input_keys':['images','questions'],'private_input_keys':['images','questions','answers','answer_types'],
        'runtime':'GPUvision_only_CPUtext_fusion_QFormer_heads_LLM_float32;CPU8interop1',
        'mode':'eval_with_autograd;NOTstochasticTRAINmode','seed':42,
        'CPU_hybrid_probability_atol':1e-4,'CPU_hybrid_answers_exact_required':True,
        'same_public_forward_sigmoid_exact_required':True,'fresh_process_recorded_results_exact_required':True,
        'optimizer_updates':0,'threshold_searches':0,'no_reserved_role_inference':True,'new_quality_training_released':False})
    print('Prepared fixed2TRAINfalse-negatives plus2TRAINtrue-negative controls, zero models')


def measure():
    import torch
    import torch.nn.functional as F
    from src.data import build_dataset,collate_fn
    from src.train import build_split
    from src.models.bonevqa import BoneVQAModel
    from src.utils import set_seed
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    check_environment();torch.set_num_interop_threads(1);p=read(OUT/'protocol.json')
    for path,digest in p['dependencies_sha256'].items():assert sha(path)==digest,path
    c=read(p['config']);ds=build_split(build_dataset,c['data']['dataset'],'train',None,True,c['data']['loader_kwargs'])
    b=collate_fn([ds[i] for i in p['indices']]);assert b['image_ids']==p['image_ids']
    private={k:b[k] for k in p['private_input_keys']};public={k:b[k] for k in p['public_input_keys']}
    target=torch.tensor([float(a=='yes') for a in b['answers']]);results={}
    def hashes(model):return {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}
    def cosine(a,b):
        den=float(a.norm()*b.norm());return float((a.flatten()*b.flatten()).sum())/den if den else None
    for mode,info in p['targets'].items():
        assert sha(info['path'])==info['sha256'];set_seed(42)
        model=BoneVQAModel.load(info['path'],device='cpu').eval();before=hashes(model)
        assert not any(model.cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt'))
        assert model.cfg['fracture_yes_threshold']==.5 and model.cfg.get('fracture_platt_slope') is None
        with torch.no_grad():cpu=model.predict_batch(public)
        original=model.visual.forward
        for m in (model.visual.trunk,model.visual.head,model.visual.proj_tokens):m.to('mps')
        def visual_forward(pixel):return {k:v.to('cpu') if torch.is_tensor(v) else v for k,v in original(pixel.to('mps')).items()}
        model.visual.forward=visual_forward
        assert next(model.llm.parameters()).device.type==next(model.text.parameters()).device.type=='cpu'
        capture={}
        def fracture_hook(module,args,output):capture.update(x=args[0],logit=output)
        def closed_hook(module,args,output):capture['closed']=output
        handles=[model.fracture_head.register_forward_hook(fracture_hook),model.closed_head.register_forward_hook(closed_hook)]
        out=model(private,stage=3)
        for h in handles:h.remove()
        x,logit=capture['x'],capture['logit']
        weighted=F.binary_cross_entropy_with_logits(logit.float(),target,pos_weight=torch.tensor(2110/458))
        ce=F.cross_entropy(capture['closed'].float(),torch.tensor([model.answer_to_idx[a] for a in b['answers']]),reduction='none').mean()
        assert float((weighted.detach()-out['loss_fracture']).abs())==0
        assert float((ce.detach()-out['loss_cls']).abs())==0
        head=list(model.fracture_head.parameters())
        tensors=[x,logit]+head
        grads={name:torch.autograd.grad(loss,tensors,retain_graph=True,allow_unused=True) for name,loss in [('BCE',weighted),('CE',ce),('total',out['loss'])]}
        assert all(g is None or torch.isfinite(g).all() for values in grads.values() for g in values)
        head_error=max(float((a-b).abs().max()) for a,b in zip(grads['BCE'][2:],grads['total'][2:]))
        assert head_error==0
        public_capture={}
        def public_hook(module,args,output):public_capture['logit']=output.detach()
        h=model.fracture_head.register_forward_hook(public_hook)
        with torch.no_grad():hybrid=model.predict_batch(public)
        h.remove()
        assert torch.sigmoid(public_capture['logit']).tolist()==[v['p_yes'] for v in hybrid]
        errors=[abs(a['p_yes']-v['p_yes']) for a,v in zip(cpu,hybrid)]
        assert max(errors)<=p['CPU_hybrid_probability_atol'] and [v['answer'] for v in cpu]==[v['answer'] for v in hybrid]
        closed_soft=torch.softmax(capture['closed'].detach().float(),dim=-1)
        pair=closed_soft[:,[model.answer_to_idx['yes'],model.answer_to_idx['no']]]
        closed_yes=(pair[:,0]/pair.sum(dim=1)).tolist()
        rows=[]
        for i in range(4):
            rows.append({'image_id':p['image_ids'][i],'gt':b['answers'][i],
                'CPU_p_yes':cpu[i]['p_yes'],'hybrid_p_yes':hybrid[i]['p_yes'],
                'closed_head_yes_no_probability':closed_yes[i],
                'closed_head_argmax_answer':model.answer_vocab[int(closed_soft[i].argmax())],
                'BCE_logit_gradient':float(grads['BCE'][1][i]),
                'BCE_x_f_gradient_norm':float(grads['BCE'][0][i].norm()),
                'CE_x_f_gradient_norm':float(grads['CE'][0][i].norm()),
                'BCE_CE_x_f_gradient_cosine':cosine(grads['BCE'][0][i],grads['CE'][0][i])})
        assert hashes(model)==before and sha(info['path'])==info['sha256']
        results[mode]={'rows':rows,'weighted_BCE':float(weighted.detach()),'CE':float(ce.detach()),
            'aggregate_BCE_CE_x_f_cosine':cosine(grads['BCE'][0],grads['CE'][0]),
            'head_total_vs_manualBCE_gradient_max_error':head_error,'CPU_hybrid_max_probability_error':max(errors),
            'all263weights_and_checkpoint_unchanged':True,'same_public_forward_sigmoid_exact':True}
        del model,out,grads,capture,tensors,head,weighted,ce;gc.collect();torch.mps.empty_cache()
    return results


def replay():
    assert measure()==read(OUT/'results.json')
    dump(OUT/'fresh_process_replay.json',{'passed':True,'recorded_results_exact':True,'optimizer_updates':0})


def run():
    assert not (OUT/'progress.json').exists();dump(OUT/'progress.json',{'status':'running'});begin=time.perf_counter()
    try:
        dump(OUT/'results.json',measure());subprocess.run([sys.executable,str(Path(__file__).resolve()),'replay'],check=True)
        report={'scope':'Two nativeTRAINFN+twoTN diagnosis, selectionbiasnotqualitycomparison',
            'protocol_sha256':sha(OUT/'protocol.json'),'results_sha256':sha(OUT/'results.json'),
            'replay_sha256':sha(OUT/'fresh_process_replay.json'),'elapsed_seconds':time.perf_counter()-begin,
            'optimizer_updates':0,'reserved_role_inference':False,'new_quality_training_released':False,
            'independent_artifact_audit_pending':True}
        dump(OUT/'report.json',report);dump(OUT/'progress.json',{'status':'replayed_pending_independent_audit','report_sha256':sha(OUT/'report.json')})
    except Exception as error:
        dump(OUT/'progress.json',{'status':'failed_preserved_no_automatic_retry','failure':repr(error)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','replay'])
    {'prepare':prepare,'run':run,'replay':replay}[parser.parse_args().mode]()
