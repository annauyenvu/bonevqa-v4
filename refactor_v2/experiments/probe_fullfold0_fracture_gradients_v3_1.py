"""Four-component TRAIN gradient probe, eval-mode autograd, ZERO updates.

Tests routing/algebra and measures local loss-gradient interaction. It is not
training-mode stochastic coverage, quality evidence or a full-TRAIN conclusion.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
PARENT=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
OUT=ROOT/'refactor_v2/reports/fullfold0_fracture_gradient_probe_v3_1'
MODES=('token_mean','sequence_uniform')
QUESTION='Is there a fracture in this X-ray?'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare():
    assert not OUT.exists()
    parent=read(PARENT/'protocol.json')
    cfg=read(parent['configs'][MODES[0]]['path'])
    train=Path(cfg['data']['loader_kwargs']['sealed_records_by_split']['train'])
    rows=read(train);batching=cfg['train']['component_batching']
    manifest=Path(batching['manifest']);seal=Path(batching['seal'])
    assert read(seal)['artifacts'][manifest.relative_to(ROOT).as_posix()]==sha(manifest)
    lookup={image:group for group,images in enumerate(read(manifest)['components']) for image in images}
    chosen=[];used=set()
    for label in ('no','yes','no','yes'):
        index=next(i for i,r in enumerate(rows) if r['question']==QUESTION
                   and r['answer']==label and lookup[r['image_id']] not in used)
        chosen.append(index);used.add(lookup[rows[index]['image_id']])
    deps={str(Path(__file__).resolve()):sha(__file__),str(train):sha(train),
          str(manifest):sha(manifest),str(seal):sha(seal)}
    deps.update(read(PARENT/'runtime_binding.json')['dependency_sha256'])
    fit=ROOT/'refactor_v2/reports/fullfold0_TRAIN_canonical_diagnostic_v3_1'
    assert read(fit/'independent_artifact_audit.json')['passed']
    assert read(fit/'independent_artifact_audit.json')['report_sha256']==sha(fit/'report.json')
    for p in [fit/'report.json',fit/'independent_artifact_audit.json']:deps[str(p)]=sha(p)
    targets={}
    for mode in MODES:
        c=parent['configs'][mode];cfgmode=read(c['path']);assert sha(c['path'])==c['sha256']
        path=ROOT/cfgmode['checkpoint_dir']/cfgmode['run_name']/'last.pt'
        digest=read(PARENT/(mode+'_completion.json'))['last_sha256'];assert sha(path)==digest
        targets[mode]={'path':str(path),'sha256':digest,'config':c}
    OUT.mkdir()
    dump(OUT/'protocol.json',{'scope':'Tiny post-diagnostic TRAIN-only eval-mode gradient algebra/routing probe',
        'dependencies_sha256':deps,'TRAIN_manifest':str(train),'selected_indices':chosen,
        'image_ids':[rows[i]['image_id'] for i in chosen],
        'selection':'First canonical no/yes/no/yes on unused known components in TRAIN order; no prediction selection',
        'targets':targets,'examples':4,'components':4,'positives':2,'negatives':2,
        'model_mode':'eval with autograd enabled; NOT stochastic TRAIN-mode gradient audit',
        'private_forward_keys':['images','questions','answers','answer_types'],
        'public_predict_keys':['images','questions'],'raw_threshold':.5,
        'measurements':['Manual weighted BCE and closed CE vs frozen forward',
            'Total fracture-head gradients equal manual weighted BCE gradients',
            'BCE/CE/total gradients on shared x_f, QFormer query and first visual LoRA-B',
            'Local gradient cosine/norms, positive and negative logit-gradient signs',
            'Public p_yes equals manual fracture-head sigmoid, all263weights unchanged'],
        'optimizer_updates':0,'threshold_searches':0,'new_weights_saved':False,
        'calibration_test_or_new_modelval_inference':False,'fresh_process_exact_replay_required':True,
        'no_automatic_retry':True,'new_quality_training_released':False})
    print('Prepared four-component gradient probe, zero models/updates')


def validate():
    p=read(OUT/'protocol.json')
    for path,digest in p['dependencies_sha256'].items():assert sha(path)==digest,path
    for info in p['targets'].values():assert sha(info['path'])==info['sha256']
    preflight=read(OUT/'independent_preflight.json')
    assert preflight['passed'] and preflight['protocol_sha256']==sha(OUT/'protocol.json')
    return p


def probe(seed):
    import torch
    import torch.nn.functional as F
    from src.data import build_dataset,collate_fn
    from src.train import build_split
    from src.models.bonevqa import BoneVQAModel
    from src.utils import set_seed
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    check_environment();plan=validate()
    c=read(plan['targets'][MODES[0]]['config']['path'])
    ds=build_split(build_dataset,c['data']['dataset'],'train',None,True,c['data']['loader_kwargs'])
    batch=collate_fn([ds[i] for i in plan['selected_indices']])
    assert batch['image_ids']==plan['image_ids']
    private={k:batch[k] for k in plan['private_forward_keys']}
    target=torch.tensor([float(a=='yes') for a in batch['answers']])
    def digest(model):return {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}
    def summary(g):
        if g is None:return {'connected':False,'norm':None,'max_abs':None}
        assert torch.isfinite(g).all()
        return {'connected':True,'norm':float(g.norm()),'max_abs':float(g.abs().max())}
    def cosine(a,b):
        if a is None or b is None:return None
        den=float(a.norm()*b.norm())
        return float((a.flatten()*b.flatten()).sum())/den if den>0 else None
    results={}
    for mode,info in plan['targets'].items():
        set_seed(seed);model=BoneVQAModel.load(info['path'],device='cpu').eval();before=digest(model)
        assert len(before)==263 and model.cfg['fracture_positive_weight']==2110/458
        assert model.cfg['fracture_yes_threshold']==.5 and model.cfg.get('fracture_platt_slope') is None
        assert not any(model.cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt'))
        captured={}
        def fracture_hook(module,args,out):captured.update(x_f=args[0],fracture_logit=out)
        def closed_hook(module,args,out):captured['closed_logits']=out
        handles=[model.fracture_head.register_forward_hook(fracture_hook),model.closed_head.register_forward_hook(closed_hook)]
        out=model(private,stage=3)
        for handle in handles:handle.remove()
        logit=captured['fracture_logit'];x=captured['x_f']
        bce=F.binary_cross_entropy_with_logits(logit.float(),target,pos_weight=torch.tensor(2110/458))
        labels=torch.tensor([model.answer_to_idx[a] for a in batch['answers']])
        ce=F.cross_entropy(captured['closed_logits'].float(),labels)
        assert float((bce.detach()-out['loss_fracture']).abs())==0
        assert float((ce.detach()-out['loss_cls']).abs())==0
        assert abs(float(out['loss_answer'].detach()-ce.detach()-bce.detach()))<1e-6
        params=dict(model.named_parameters())
        head_names=[n for n,p in params.items() if n.startswith('fracture_head.') and p.requires_grad]
        selected=head_names+['qformer.query',next(n for n,p in params.items() if n.startswith('visual.') and 'lora_B' in n and p.requires_grad)]
        tensors=[x,logit]+[params[n] for n in selected]
        gradients={key:torch.autograd.grad(loss,tensors,retain_graph=True,allow_unused=True)
                   for key,loss in [('BCE',bce),('CE',ce),('total',out['loss'])]}
        errors={n:float((gradients['BCE'][i+2]-gradients['total'][i+2]).abs().max()) for i,n in enumerate(head_names)}
        assert all(v<1e-6 for v in errors.values())
        expected=(torch.sigmoid(logit.detach())-target)*torch.where(target>0,torch.tensor(2110/458),torch.tensor(1.))/4
        assert torch.allclose(gradients['BCE'][1],expected,atol=1e-7,rtol=1e-6)
        assert all(float(g)<0 if t else float(g)>0 for g,t in zip(gradients['BCE'][1],target))
        manual_probs=torch.sigmoid(logit.detach()).tolist()
        gradient_report={name:{k:summary(v[i]) for k,v in gradients.items()} for i,name in enumerate(['shared_x_f','fracture_logit']+selected)}
        local_cosine={name:cosine(gradients['BCE'][i],gradients['CE'][i]) for i,name in enumerate(['shared_x_f','fracture_logit']+selected)}
        with torch.no_grad():predictions=model.predict_batch({'images':batch['images'],'questions':batch['questions']})
        public_probs=[p['p_yes'] for p in predictions]
        assert manual_probs==public_probs
        assert digest(model)==before and sha(info['path'])==info['sha256']
        results[mode]={'BCE':float(bce.detach()),'CE':float(ce.detach()),'total_loss':float(out['loss'].detach()),
            'BCE_logit_gradient':gradients['BCE'][1].tolist(),'head_total_vs_manualBCE_gradient_max_errors':errors,
            'gradients':gradient_report,'BCE_CE_cosines':local_cosine,
            'public_probabilities':public_probs,'public_sigmoid_exact':True,'all263weights_unchanged':True,
            'checkpoint_sha256':info['sha256']}
        del model,out,bce,ce,captured,gradients,tensors,params;gc.collect()
    return results


def replay():
    assert probe(90211)==read(OUT/'results.json')
    dump(OUT/'fresh_process_replay.json',{'passed':True,'all_values_gradients_public_probabilities_exact':True,
        'constructor_seed':90211,'optimizer_updates':0})


def run():
    validate();assert not (OUT/'progress.json').exists()
    dump(OUT/'progress.json',{'status':'running'});started=time.perf_counter()
    try:
        results=probe(90210);dump(OUT/'results.json',results)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'replay'],check=True)
        report={'scope':'Four TRAIN examples, eval-mode autograd, no efficacy or full-TRAIN conflict claim',
            'protocol_sha256':sha(OUT/'protocol.json'),'results_sha256':sha(OUT/'results.json'),
            'replay_sha256':sha(OUT/'fresh_process_replay.json'),'elapsed_seconds':time.perf_counter()-started,
            'optimizer_updates':0,'new_weights_saved':False,'reserved_role_inference':False,
            'independent_artifact_audit_pending':True,'new_quality_training_released':False}
        dump(OUT/'report.json',report)
        dump(OUT/'progress.json',{'status':'replayed_pending_independent_artifact_audit','report_sha256':sha(OUT/'report.json')})
    except Exception as error:
        dump(OUT/'progress.json',{'status':'failed_preserved_no_automatic_retry','failure':repr(error)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','replay'])
    {'prepare':prepare,'run':run,'replay':replay}[parser.parse_args().mode]()
