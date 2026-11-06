"""Two actual native-asset route updates, all source-sized adapter groups.

Disposable technical SGD, not source quality optimizer or pretraining evidence.
"""
import argparse
import gc
import json
from pathlib import Path
import resource
import shutil
import time
import torch
from experiments.favp_pilot_route_v1 import PilotRoute as TrainableRoute, ROOT, optimizer_for, pilot_batch
from experiments.lapa_integrated_gate_v1 import batch_data, digest
from experiments.prepare_lapa_pilot_v1 import sha

OUT = ROOT / 'reports/favp_pilot_optimizer_gate_v1'


def main(replay=False):
    if not replay:
        OUT.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    started = time.perf_counter()
    model = TrainableRoute()
    groups = model.groups()
    mutable_names = {n for n,p in model.named_parameters(remove_duplicate=False) if p.requires_grad}
    inventory = {g:sum(p.numel() for n,p in ps) for g,ps in groups.items()}
    (OUT/'inventory.json').write_text(json.dumps({'groups':inventory,'audit':model.audit},indent=2)) if not replay else None
    frozen_versions = {n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    before = {g:digest(ps) for g,ps in groups.items()}
    batch, answers = pilot_batch()
    sources = [Path(__file__), ROOT/'experiments/favp_trainable_route_v3.py', ROOT/'experiments/favp_pilot_route_v1.py',
        ROOT/'experiments/favp_generation_compat_v1.py',ROOT/'experiments/favp_circle_cache_v1.py',
        ROOT/'reports/favp_pilot_inputs_v1/selection_and_schedules.json',ROOT/'reports/favp_pilot_inputs_v1/public_validation_cache.json',
        ROOT/'experiments/check_favp_cpu_compute_v1.py', ROOT/'experiments/favp_vicuna_loader_v3.py',
        ROOT/'experiments/favp_visual_bridge_v2.py', ROOT/'experiments/check_favp_projector_update_v1.py',
        ROOT/'reports/favp_train_proposal_census_v1/receipt.json']
    bindings = {str(p):sha(p) for p in sources}
    if replay:
        saved = torch.load(OUT/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert saved['sources']==bindings and saved['initial_group_digests']==before
        assert set(saved['mutable'])==mutable_names
        full = model.state_dict()
        full.update(saved['mutable'])
        model.load_state_dict(full,strict=True)
        del full
        model.eval()
        with torch.no_grad():
            result = model(batch['images'][0],batch['questions'][0],answers[0])
        assert torch.equal(result.logits,saved['logits']) and torch.equal(result.loss,saved['loss'])
        (OUT/'replay.json').write_text(json.dumps({'strict_mutable_reload':True,'exact_logits':True,'exact_loss':True,
            'checkpoint_sha256':sha(OUT/'checkpoint.pt'),'seconds':time.perf_counter()-started},indent=2))
        return
    # Small fresh technical SGD budget; no disposable prior weights are loaded.
    opt,scheduler = optimizer_for(model)
    steps=[]
    for step in range(2):
        assert time.perf_counter()-started<1800, 'Technical gate time budget exceeded'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 23*1024**3, 'Observed peak RAM guard'
        torch.manual_seed(42+step)
        model.train()
        opt.zero_grad(set_to_none=True)
        tick=time.perf_counter()
        result=model(batch['images'][step],batch['questions'][step],answers[step])
        assert torch.isfinite(result.loss)
        result.loss.backward()
        stats={}
        for name,params in groups.items():
            gradients=[p.grad for n,p in params if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients),name
            stats[name]=max(float(g.abs().max()) for g in gradients)
            assert stats[name]>0,name
        assert all(p.grad is None and p._version==frozen_versions[n] for n,p in model.named_parameters() if not p.requires_grad)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        loss=float(result.loss.detach())
        learning_rate=opt.param_groups[0]['lr']
        opt.step()
        scheduler.step()
        del result
        gc.collect()
        steps.append({'step':step+1,'image_id':batch['image_ids'][step],'loss':loss,'group_grad_max':stats,'seconds':time.perf_counter()-tick,'learning_rate':learning_rate,'scheduler_step':scheduler.last_epoch})
        (OUT/f'step{step+1}.json').write_text(json.dumps(steps[-1],indent=2))
    changed={g:digest(ps)!=before[g] for g,ps in groups.items()}
    assert all(changed.values())
    for ps in (groups['EVA_LoRA'],groups['LM_LoRA']):
        assert any(p.grad is not None and p.grad.abs().max()>0 for n,p in ps if n.endswith('A.weight'))
        assert any(p.grad is not None and p.grad.abs().max()>0 for n,p in ps if n.endswith('B.weight'))
    assert all(p._version==frozen_versions[n] for n,p in model.named_parameters() if not p.requires_grad)
    opt.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        result=model(batch['images'][0],batch['questions'][0],answers[0])
    assert torch.isfinite(result.loss) and torch.isfinite(result.logits).all()
    required=sum(t.numel()*t.element_size() for n,t in model.state_dict().items() if n in mutable_names)
    assert shutil.disk_usage(OUT).free-required>8*1024**3
    torch.save({'mutable':{n:t for n,t in model.state_dict().items() if n in mutable_names},
        'initial_group_digests':before,'sources':bindings,'logits':result.logits,'loss':result.loss},OUT/'checkpoint.pt')
    (OUT/'update.json').write_text(json.dumps({'updates':2,'steps':steps,'changed_groups':changed,
        'frozen_versions_unchanged':True,'frozen_gradients_absent':True,'audit':model.audit,
        'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'seconds':time.perf_counter()-started,
        'scope':'actual image→EVA-LoRA→trainable QFormer→new projection→Vicuna-LoRA; census-verified original fallback; disposable scheduled AdamW technical gate, not quality/source stage handoff',
        'quality_released':False},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    try:
        main(args.replay)
    except Exception as exc:
        if OUT.exists():
            with (OUT/('replay_failure.json' if args.replay else 'failure.json')).open('x') as f:
                json.dump({'error':repr(exc)},f)
        raise



