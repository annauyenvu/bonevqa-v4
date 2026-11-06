"""Two actual accumulated updates/arm, compact strict checkpoint replay."""
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.lapa_native_pilot_v2 import SourceOptimizerPilot, ARMS
from experiments.lapa_integrated_gate_v1 import REPORTS,digest
from experiments.prepare_lapa_pilot_v1 import sha
from experiments.lapa_reference_endpoint_v1 import reference_loss
from src.data import build_dataset,collate_fn
from src.train import build_split
from src.utils import normalize_answer

ROOT=REPORTS/'lapa_native_scheduled_gates_v1'


def initialize(arm):
    torch.manual_seed(42)
    return SourceOptimizerPilot(arm)


def main():
    torch.set_num_threads(8);arm=sys.argv[1];assert arm in ARMS
    replay='--replay' in sys.argv;out=ROOT/arm
    if not replay:out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();m=initialize(arm)
    cfg=json.loads((REPORTS/'step4_localization_v3_1/B.json').read_text())
    ds=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
    schedule=json.loads((REPORTS/'step4_localization_v3_1/schedules.json').read_text())[0][:8]
    trainable={n for n,p in m.named_parameters(remove_duplicate=False) if p.requires_grad}
    initial=digest(m.state_dict().items())
    fixed=lambda:digest((n,t) for n,t in m.state_dict().items() if n not in trainable)
    if replay:
        saved=torch.load(out/'trainable_state.pt',map_location='cpu')
        assert initial==saved['initial_digest'] and fixed()==saved['fixed_digest']
        assert set(saved['trainable'])==trainable
        state=m.state_dict();state.update(saved['trainable']);m.load_state_dict(state,strict=True)
        assert digest(m.state_dict().items())==saved['full_final_digest']
        m.eval();actual=[]
        with torch.no_grad():
            for indices in schedule:
                b=collate_fn([ds[i] for i in indices]);actual.append(m(b['images'],b['questions'])[0])
        assert all(torch.equal(a,b) for a,b in zip(actual,saved['logits']))
        (out/'replay.json').write_text(json.dumps({'strict_reconstruction':True,'full_state_digest_exact':True,'eight_microbatch_logits_exact':True}));return
    required=sum(p.numel()*p.element_size() for p in m.parameters() if p.requires_grad)
    assert shutil.disk_usage(ROOT).free-required > 8*1024**3,'Insufficient 8GiB reserve'
    frozen=fixed()
    prefixes={'GAT':'slake_pf.pf.GAN_layer.','PF_attention':'slake_pf.pf.cross_attn_kw.',
              'graph_pool':'slake_pf.graph_pool.','combined_pool':'slake_pf.combined_pool.',
              'GM':'backbone.reference.prompt.generation.','LF':'backbone.reference.prompt.LF.',
              'text_MM':'backbone.reference.text_layers.','image_MM':'backbone.reference.image_layers.','head':'head.'}
    group=lambda prefix:[(n,p) for n,p in m.named_parameters() if n.startswith(prefix) and p.requires_grad]
    prefixes={k:p for k,p in prefixes.items() if group(p)}
    before={k:digest(group(p)) for k,p in prefixes.items()}
    lora=lambda:[(n,p) for n,p in m.named_parameters() if 'lora_' in n]
    before_lora=None
    length=max(len(m.text.tokenizer.encode(a,add_special_tokens=True)) for a in m.vocab)
    m.train();opt=m.optimizer();scheduler=m.scheduler(opt);steps=[]
    torch.manual_seed(43)
    for update in range(2):
        tick=time.perf_counter();opt.zero_grad();losses=[]
        for indices in schedule[update*4:update*4+4]:
            records=[ds[i] for i in indices];b=collate_fn(records)
            answers=[normalize_answer(r['answer']) for r in records]
            targets=F.one_hot(torch.tensor([m.vocab.index(a) for a in answers]),41).float()
            ids=m.text.tokenizer(answers,padding='max_length',max_length=length,truncation=False,return_tensors='pt')['input_ids']
            with torch.no_grad():words=m.text.model.get_input_embeddings()(ids)
            logits,g=m(b['images'],b['questions'])
            loss=reference_loss(logits,targets,g,words,m.eta)['total'];assert torch.isfinite(loss)
            (loss/4).backward();losses.append(float(loss))
        gradient_max={k:max(float(p.grad.abs().max()) for _,p in group(prefix) if p.grad is not None) for k,prefix in prefixes.items()}
        assert all(v>0 for v in gradient_max.values())
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);assert torch.isfinite(norm)
        opt.step();scheduler.step();steps.append({'losses':losses,'gradient_max':gradient_max,'seconds':time.perf_counter()-tick})
    changed={k:digest(group(p))!=before[k] for k,p in prefixes.items()}
    assert all(changed.values()), changed
    assert frozen==fixed(), 'fixed state changed'
    m.eval();logits=[]
    with torch.no_grad():
        for indices in schedule:
            b=collate_fn([ds[i] for i in indices]);logits.append(m(b['images'],b['questions'])[0])
    final=digest(m.state_dict().items())
    torch.save({'trainable':{n:t.detach() for n,t in m.state_dict().items() if n in trainable},
                'initial_digest':initial,'fixed_digest':frozen,'full_final_digest':final,'logits':logits},out/'trainable_state.pt')
    del m,opt,scheduler;gc.collect()
    subprocess.run([sys.executable,str(Path(__file__).resolve()),arm,'--replay'],check=True)
    receipt={'arm':arm,'initial_digest':initial,'updates':2,'microbatches':schedule,'steps':steps,'changed_groups':changed,
             'fixed_state_unchanged':True,'checkpoint_bytes':(out/'trainable_state.pt').stat().st_size,
             'replay':json.loads((out/'replay.json').read_text()),'seconds':time.perf_counter()-start,'quality_released':False,
             'checkpoint_retention':'temporary checkpoint removed only after successful fresh replay; full-state digest retained',
             'final_state_digest':final,
             'source_hashes':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('lapa_native_pilot_v2.py')]}}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2))
    (out/'trainable_state.pt').unlink()
    print(json.dumps({'arm':arm,'passed':True,'seconds':receipt['seconds']}))


if __name__=='__main__':
    try:main()
    except Exception as error:
        out=ROOT/sys.argv[1]
        if out.exists():
            with (out/('replay_failure.json' if '--replay' in sys.argv else 'failure.json')).open('x') as f:json.dump({'error':repr(error)},f)
        raise


