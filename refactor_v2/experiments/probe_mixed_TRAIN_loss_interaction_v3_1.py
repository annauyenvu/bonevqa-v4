"""One deterministic mixed forward/checkpoint; native/uniform LM and fracture
gradients on shared parameters. Six post-hoc TRAIN examples, zero updates.
"""
import argparse
import gc
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from experiments.probe_error_bearing_TRAIN_v3_1 import read,sha,dump,ROOT

OUT=ROOT/'refactor_v2/reports/mixed_TRAIN_loss_interaction_v3_1'
HARD=ROOT/'refactor_v2/reports/error_bearing_TRAIN_probe_v3_1'
EASY=ROOT/'refactor_v2/reports/fullfold0_uniform_lengths_v3_1'


def prepare():
    assert not OUT.exists();h=read(HARD/'protocol.json');rows=read(h['TRAIN_manifest'])
    assert read(HARD/'independent_artifact_review.json')['passed']
    indices=h['indices']+[i for i in read(EASY/'protocol.json')['TRAIN_indices'] if rows[i]['answer_type']=='OPEN']
    cfg=read(h['config']);b=cfg['train']['component_batching']
    lookup={image:g for g,images in enumerate(read(b['manifest'])['components']) for image in images}
    assert len(indices)==6 and len({lookup[rows[i]['image_id']] for i in indices})==6
    assert [rows[i]['answer_type'] for i in indices]==['CLOSED']*4+['OPEN']*2
    deps={**h['dependencies_sha256'],str(Path(__file__).resolve()):sha(__file__)}
    for path in [HARD/'protocol.json',HARD/'report.json',HARD/'independent_artifact_review.json',EASY/'protocol.json',ROOT/'refactor_v2/experiments/probe_error_bearing_TRAIN_v3_1.py']:
        deps[str(path)]=sha(path)
    OUT.mkdir();dump(OUT/'protocol.json',{'scope':'Post-hoc sixTRAIN shared-parameter local gradient interaction, not quality/causal benefit',
        'dependencies_sha256':deps,'config':h['config'],'TRAIN_manifest':h['TRAIN_manifest'],
        'targets':h['targets'],'indices':indices,'image_ids':[rows[i]['image_id'] for i in indices],
        'selection':'Prior2nativeFN2TNpluspriorfixedshort/longOPENrows,all6knowncomponentsdistinct',
        'loss_rules':'Same frozen forward/checkpoint:actual native LM vs sequence_uniform from SAMElabels/logits using exact existing sequence_losses;TRAINweightsallones',
        'mode':'eval_autograd_dropoutoff;manual loss algebra same as TRAIN adapter, not stochastic training-mode audit',
        'runtime':'visionMPS_restCPUfloat32_CPU8interop1','optimizer_updates':0,
        'reserved_role_inference':False,'new_quality_training_released':False,
        'fresh_process_recorded_summary_replay_exact_required':True,
        'do_not':'Interpretdisjointperrowx_fgradientdotproductsascompetition;measureSHAREDparametergradients',
        'no_automatic_retry':True})
    print('Prepared6TRAINcomponents4CLOSED2OPEN, sameforward loss contrasts, zero models')


def measure():
    import torch
    import torch.nn.functional as F
    from src.data import build_dataset,collate_fn
    from src.train import build_split
    from src.models.bonevqa import BoneVQAModel
    from src.utils import set_seed
    from experiments.sequence_answer_loss_candidate_v3_1 import sequence_losses
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    check_environment();torch.set_num_interop_threads(1);p=read(OUT/'protocol.json')
    for path,digest in p['dependencies_sha256'].items():assert sha(path)==digest,path
    c=read(p['config']);ds=build_split(build_dataset,c['data']['dataset'],'train',None,True,c['data']['loader_kwargs'])
    b=collate_fn([ds[i] for i in p['indices']]);assert b['image_ids']==p['image_ids']
    private={k:b[k] for k in ('images','questions','answers','answer_types')};results={}
    def hashes(model):return {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}
    def stats(g):
        if g is None:return {'connected':False,'norm':None,'max_abs':None}
        assert torch.isfinite(g).all();return {'connected':True,'norm':float(g.norm().cpu()),'max_abs':float(g.abs().max().cpu())}
    def cosine(a,v):
        if a is None or v is None:return None
        denominator=float((a.norm()*v.norm()).cpu())
        return float((a.flatten()*v.flatten()).sum().cpu())/denominator if denominator else None
    for mode,info in p['targets'].items():
        assert sha(info['path'])==info['sha256'];set_seed(42)
        model=BoneVQAModel.load(info['path'],device='cpu').eval();before=hashes(model)
        assert not any(model.cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt'))
        original=model.visual.forward
        for m in (model.visual.trunk,model.visual.head,model.visual.proj_tokens):m.to('mps')
        def visual_forward(pixel):return {k:v.to('cpu') if torch.is_tensor(v) else v for k,v in original(pixel.to('mps')).items()}
        model.visual.forward=visual_forward;assert next(model.llm.parameters()).device.type=='cpu'
        capture={}
        def llm_hook(module,args,kwargs,output):capture.update(labels=kwargs['labels'],logits=output.logits,native_loss=output.loss)
        def fracture_hook(module,args,output):capture['fracture']=output
        def closed_hook(module,args,output):capture['closed']=output
        handles=[model.llm.register_forward_hook(llm_hook,with_kwargs=True),model.fracture_head.register_forward_hook(fracture_hook),model.closed_head.register_forward_hook(closed_hook)]
        out=model(private,stage=3)
        for handle in handles:handle.remove()
        lm=sequence_losses(capture['logits'],capture['labels'],[1.,1.])
        counts=capture['labels'][:,1:].ne(-100).sum(dim=1).tolist();assert len(counts)==2 and counts[0]!=counts[1]
        bce=F.binary_cross_entropy_with_logits(capture['fracture'][:4].float(),torch.tensor([1.,0.,1.,0.]),pos_weight=torch.tensor(2110/458))
        ce=F.cross_entropy(capture['closed'][:4].float(),torch.tensor([model.answer_to_idx[a] for a in b['answers'][:4]]),reduction='none').mean()
        assert float((bce.detach()-out['loss_fracture']).abs())==0 and float((ce.detach()-out['loss_cls']).abs())==0
        assert abs(float(lm['token_mean'].detach()-capture['native_loss'].detach()))<=1e-6
        params=dict(model.named_parameters());names=['fusion.proj_i.0.weight','fusion.out_norm.weight','qformer.query','prefix_proj.0.weight','fracture_head.net.3.weight',next(n for n in params if n.startswith('visual.') and 'lora_B' in n),next(n for n in params if n.startswith('llm.') and 'lora_B' in n)]
        tensors=[params[n] for n in names]
        objectives={'fracture_BCE':bce,'closed_CE':ce,'actual_native_LM':capture['native_loss'],'token_mean_LM':lm['token_mean'],'sequence_uniform_LM':lm['sequence_uniform']}
        gradients={name:torch.autograd.grad(loss,tensors,retain_graph=True,allow_unused=True) for name,loss in objectives.items()}
        equivalent={n:(float((gradients['actual_native_LM'][i]-gradients['token_mean_LM'][i]).abs().max().cpu()) if gradients['actual_native_LM'][i] is not None else None) for i,n in enumerate(names)}
        assert all(v is None or v<=1e-6 for v in equivalent.values())
        parameter_report={}
        for i,name in enumerate(names):
            parameter_report[name]={'gradients':{k:stats(values[i]) for k,values in gradients.items()},
                'fracture_vs_native_LM_cosine':cosine(gradients['fracture_BCE'][i],gradients['token_mean_LM'][i]),
                'fracture_vs_uniform_LM_cosine':cosine(gradients['fracture_BCE'][i],gradients['sequence_uniform_LM'][i]),
                'native_vs_uniform_LM_cosine':cosine(gradients['token_mean_LM'][i],gradients['sequence_uniform_LM'][i])}
        assert hashes(model)==before and sha(info['path'])==info['sha256']
        results[mode]={'losses':{k:float(v.detach()) for k,v in objectives.items()},'OPEN_supervised_token_counts':counts,
            'actual_native_vs_manual_tokenmean_gradient_max_errors':equivalent,'parameters':parameter_report,
            'all263weights_and_checkpoint_unchanged':True}
        del model,out,capture,gradients,tensors,params,objectives,lm,bce,ce;gc.collect();torch.mps.empty_cache()
    return results


def replay():
    assert measure()==read(OUT/'results.json');dump(OUT/'fresh_process_replay.json',{'passed':True,'recorded_losses_gradientsummaries_cosines_exact':True,'optimizer_updates':0})


def run():
    assert not (OUT/'progress.json').exists();dump(OUT/'progress.json',{'status':'running'});begin=time.perf_counter()
    try:
        dump(OUT/'results.json',measure());subprocess.run([sys.executable,str(Path(__file__).resolve()),'replay'],check=True)
        report={'protocol_sha256':sha(OUT/'protocol.json'),'results_sha256':sha(OUT/'results.json'),'replay_sha256':sha(OUT/'fresh_process_replay.json'),
            'elapsed_seconds':time.perf_counter()-begin,'optimizer_updates':0,'reserved_role_inference':False,'new_quality_training_released':False,'independent_artifact_audit_pending':True}
        dump(OUT/'report.json',report);dump(OUT/'progress.json',{'status':'replayed_pending_independent_audit','report_sha256':sha(OUT/'report.json')})
    except Exception as error:dump(OUT/'progress.json',{'status':'failed_preserved_no_automatic_retry','failure':repr(error)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','replay'])
    {'prepare':prepare,'run':run,'replay':replay}[parser.parse_args().mode]()
