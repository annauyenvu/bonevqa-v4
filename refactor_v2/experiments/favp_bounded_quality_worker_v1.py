"""Fixed 32-TRAIN/8-validation adapted generative pilot; verified release only."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
import torch
from experiments.favp_pilot_route_v1 import PilotRoute,ROOT,INPUTS,optimizer_for
from experiments.check_favp_public_generation_v2 import public_generate
from experiments.lapa_integrated_gate_v1 import digest
from experiments.prepare_lapa_pilot_v1 import sha
from src.data import build_dataset
from src.train import build_split
from src.utils import normalize_answer

OUT=ROOT/'reports/favp_bounded_quality_v1'
PREP=ROOT/'reports/favp_bounded_preparation_v1'


def metrics(rows):
    return {'records':len(rows),'accuracy':sum(r['normalized_answer']==r['target'] for r in rows)/len(rows),
        'unsupported':sum(r['normalized_answer'] not in ('yes','no') for r in rows),
        'empty':sum(not r['normalized_answer'] for r in rows),'truncated':sum(r['truncated'] for r in rows),
        'target_yes':sum(r['target']=='yes' for r in rows),'target_no':sum(r['target']=='no' for r in rows)}


def evaluate(model,dataset,selected,phase,deadline):
    model.eval();rows=[];started=time.perf_counter()
    with torch.no_grad():
        for selected_row in selected:
            assert time.perf_counter()<deadline,'Worker wall budget exceeded'
            r=dataset[selected_row['record_index']]
            assert r['image_id']==selected_row['image_id']
            tick=time.perf_counter()
            generated=public_generate(model,r['image'],r['question'])
            # Target is accessed only after public image/question generation.
            row={'image_id':r['image_id'],'question':r['question'],**generated,'target':normalize_answer(r['answer']),
                 'truncated':len(generated['tokens'])==20 and generated['tokens'][-1]!=model.tokenizer.eos_token_id}
            rows.append(row)
            (OUT/f'{phase}_progress.json').write_text(json.dumps({'completed':len(rows),'expected':8,
                'last_seconds':time.perf_counter()-tick,'elapsed_seconds':time.perf_counter()-started}))
            (OUT/f'{phase}_partial.json').write_text(json.dumps(rows))
    assert len(rows)==8
    return {'rows':rows,'metrics':metrics(rows),'seconds':time.perf_counter()-started}


def main(replay=False):
    started=time.perf_counter();deadline=started+14400
    release=json.loads((OUT/'execution_release.json').read_text());assert release['quality_released']
    for p,h in release['dependencies'].items():assert sha(Path(p))==h,p
    if not replay:assert not (OUT/'PRE.json').exists() and not (OUT/'LAST.pt').exists()
    else:assert not (OUT/'replay.json').exists()
    torch.set_num_threads(8);torch.manual_seed(42)
    model=PilotRoute();mutable_names={n for n,p in model.named_parameters(remove_duplicate=False) if p.requires_grad}
    shared=torch.load(PREP/'shared_PREupdate.pt',map_location='cpu',weights_only=True)
    assert shared['updates']==0
    assert shared['initial_group_digests']=={g:digest(ps) for g,ps in model.groups().items()}
    assert digest((n,t) for n,t in model.state_dict().items() if n not in mutable_names)==shared['fixed_digest']
    full=model.state_dict();full.update(shared['mutable']);model.load_state_dict(full,strict=True);del full
    assert digest((n,t) for n,t in model.state_dict().items() if n in mutable_names)==shared['mutable_digest']
    initial_digest=shared['mutable_digest'];fixed_digest=shared['fixed_digest'];del shared
    if replay:
        saved=torch.load(OUT/'LAST.pt',map_location='cpu',weights_only=True)
        assert saved['updates']==64 and saved['initial_digest']==initial_digest and saved['fixed_digest']==fixed_digest
        assert set(saved['mutable'])==mutable_names
        full=model.state_dict();full.update(saved['mutable']);model.load_state_dict(full,strict=True);del full
        assert digest((n,t) for n,t in model.state_dict().items() if n in mutable_names)==saved['mutable_digest']
        del saved
    cfg=json.loads((ROOT/'reports/step4_localization_v3_1/B.json').read_text())
    val=build_split(build_dataset,cfg['data']['dataset'],'model_val',None,True,cfg['data']['loader_kwargs'])
    selection=json.loads((INPUTS/'selection_and_schedules.json').read_text())
    if replay:
        result=evaluate(model,val,selection['subset']['model_val'],'replay',deadline)
        expected=json.loads((OUT/'epoch2.json').read_text())
        assert result['rows']==expected['rows'] and result['metrics']==expected['metrics']
        (OUT/'replay.json').write_text(json.dumps({'exact_tokens_answers_metrics':True,'records':8,'checkpoint_sha256':sha(OUT/'LAST.pt'),
            'seconds':time.perf_counter()-started},indent=2));return
    frozen_versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    before={g:digest(ps) for g,ps in model.groups().items()}
    result=evaluate(model,val,selection['subset']['model_val'],'PRE',deadline)
    result.update(updates=0,initial_digest=initial_digest)
    (OUT/'PRE.json').write_text(json.dumps(result,indent=2))
    train=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
    optimizer,scheduler=optimizer_for(model);updates=0
    for epoch,schedule in enumerate(selection['schedules'],1):
        assert len(schedule)==32
        torch.manual_seed(42+epoch);model.train();losses=[];tick=time.perf_counter()
        for index in schedule:
            assert time.perf_counter()<deadline
            assert shutil.disk_usage(OUT).free>8*1024**3
            optimizer.zero_grad(set_to_none=True)
            r=train[index];answer=normalize_answer(r['answer']);assert answer in ('yes','no')
            result=model(r['image'],r['question'],answer);assert torch.isfinite(result.loss)
            result.loss.backward()
            gradients={}
            for name,params in model.groups().items():
                gs=[p.grad for n,p in params if p.grad is not None]
                assert gs and all(torch.isfinite(g).all() for g in gs),name
                gradients[name]=max(float(g.abs().max()) for g in gs);assert gradients[name]>0
            assert all(p.grad is None and p._version==frozen_versions[n] for n,p in model.named_parameters() if not p.requires_grad)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            loss=float(result.loss.detach());lr=optimizer.param_groups[0]['lr']
            optimizer.step();scheduler.step();updates+=1;losses.append(loss)
            del result
            (OUT/'training_progress.json').write_text(json.dumps({'epoch':epoch,'updates':updates,'expected_updates':64,
                'loss':loss,'lr':lr,'grad_norm':float(norm),'group_grad_max':gradients,'elapsed_seconds':time.perf_counter()-started}))
        optimizer.zero_grad(set_to_none=True);gc.collect()
        train_seconds=time.perf_counter()-tick
        result=evaluate(model,val,selection['subset']['model_val'],f'epoch{epoch}',deadline)
        result.update(epoch=epoch,updates=updates,mean_train_loss=sum(losses)/32,train_seconds=train_seconds)
        with (OUT/f'epoch{epoch}.json').open('x') as f:json.dump(result,f,indent=2)
    assert updates==64 and scheduler.last_epoch==64
    assert fixed_digest==digest((n,t) for n,t in model.state_dict().items() if n not in mutable_names)
    changed={g:digest(ps)!=before[g] for g,ps in model.groups().items()};assert all(changed.values())
    mutable={n:t for n,t in model.state_dict().items() if n in mutable_names}
    required=sum(t.numel()*t.element_size() for t in mutable.values())
    assert shutil.disk_usage(OUT).free-required>8*1024**3
    torch.save({'mutable':mutable,'mutable_digest':digest(mutable.items()),'initial_digest':initial_digest,
        'fixed_digest':fixed_digest,'updates':64},OUT/'LAST.pt')
    (OUT/'completed.json').write_text(json.dumps({'updates':64,'epochs':2,'changed_groups':changed,
        'frozen_state_unchanged':True,'seconds':time.perf_counter()-started,'quality_scope':'bounded de-novo adapted pilot only'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    try:main(args.replay)
    except Exception as exc:
        if OUT.exists():
            with (OUT/('replay_failure.json' if args.replay else 'worker_failure.json')).open('x') as f:json.dump({'error':repr(exc)},f)
        raise
