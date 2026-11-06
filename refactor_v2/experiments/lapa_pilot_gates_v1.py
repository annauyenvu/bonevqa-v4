"""Actual scheduled microbatches, AdamW accumulation and fresh-process replay."""
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.lapa_pilot_runner_v1 import PilotRunner, ARMS
from experiments.lapa_integrated_gate_v1 import REPORTS, sha, digest
from experiments.lapa_reference_endpoint_v1 import reference_loss
from src.data import build_dataset, collate_fn
from src.train import build_split
from src.utils import normalize_answer

ROOT=REPORTS/'lapa_pilot_gates_v1'


def main():
    torch.set_num_threads(8); torch.manual_seed(42)
    arm=sys.argv[1]; assert arm in ARMS
    replay='--replay' in sys.argv
    out=ROOT/arm
    if not replay: out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    binding=json.loads((REPORTS/'lapa_port_backbone_binding_v1/cache_binding.json').read_text())
    for model in binding['models']:
        for f in model['files']: assert sha(Path(f['resolved_path']))==f['sha256']
    m=PilotRunner(arm)
    cfg=json.loads((REPORTS/'step4_localization_v3_1/B.json').read_text())
    ds=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
    schedule=json.loads((REPORTS/'step4_localization_v3_1/schedules.json').read_text())[0][:8]
    if replay:
        saved=torch.load(out/'state.pt',map_location='cpu'); m.load_state_dict(saved['state'],strict=True); m.eval()
        b=collate_fn([ds[i] for i in schedule[0]])
        with torch.no_grad(): actual,_=m(b['images'],b['questions'])
        assert torch.equal(actual,saved['logits'])
        (out/'replay.json').write_text(json.dumps({'strict':True,'exact_logits':True})); return
    checkpoint=REPORTS/'step4_localization_v3_1/B_PREtask.pt'
    release=json.loads((REPORTS/'vnext_local_tokens_v1/training_release.json').read_text())
    assert sha(checkpoint)==release['dependencies'][str(checkpoint)]
    state=torch.load(checkpoint,map_location='cpu')['state']
    supplied={k[7:]:v for k,v in state.items() if k.startswith('visual.')}
    loaded=m.visual.load_state_dict(supplied,strict=False); assert not loaded.unexpected_keys
    assert all(torch.equal(v,m.visual.state_dict()[k]) for k,v in supplied.items())
    del state,supplied
    initial=digest(m.state_dict().items())
    frozen=digest((n,p) for n,p in m.named_parameters() if not p.requires_grad)
    before={n:p.detach().clone() for n,p in m.named_parameters() if 'lora_' in n or n.startswith('head.')}
    length=max(len(m.text.tokenizer.encode(a,add_special_tokens=True)) for a in m.vocab)
    m.train(); opt=m.optimizer(); steps=[]
    for update in range(2):
        tick=time.perf_counter(); opt.zero_grad(); values=[]
        for indices in schedule[update*4:(update+1)*4]:
            records=[ds[i] for i in indices]; b=collate_fn(records)
            answers=[normalize_answer(r['answer']) for r in records]
            targets=F.one_hot(torch.tensor([m.vocab.index(a) for a in answers]),41).float()
            ids=m.text.tokenizer(answers,padding='max_length',max_length=length,truncation=False,return_tensors='pt')['input_ids']
            with torch.no_grad(): words=m.text.model.get_input_embeddings()(ids)
            logits,generated=m(b['images'],b['questions'])
            losses=reference_loss(logits,targets,generated,words,eta=m.eta)
            (losses['total']/4).backward(); values.append(losses['total'].item())
        grads=[p.grad for p in m.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        if arm=='carrier': assert all(p.grad is None for p in m.reference.prompt.parameters())
        norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.)
        assert torch.isfinite(norm); opt.step()
        steps.append({'microbatch_losses':values,'preclip_norm':norm.item(),'seconds':time.perf_counter()-tick})
    assert frozen==digest((n,p) for n,p in m.named_parameters() if not p.requires_grad)
    changed=[n for n,p in m.named_parameters() if n in before and not torch.equal(p,before[n])]
    assert any('lora_' in n for n in changed) and any(n.startswith('head.') for n in changed)
    m.eval(); b=collate_fn([ds[i] for i in schedule[0]])
    with torch.no_grad(): logits,_=m(b['images'],b['questions'])
    torch.save({'state':m.state_dict(),'logits':logits},out/'state.pt')
    del opt,m,before,grads; gc.collect()
    subprocess.run([sys.executable,str(Path(__file__).resolve()),arm,'--replay'],check=True)
    receipt={'arm':arm,'initial_state_digest':initial,'updates':2,'microbatches':schedule,'steps':steps,
             'frozen_unchanged':True,'changed_lora_head_tensors':changed,'replay':json.loads((out/'replay.json').read_text()),
             'seconds':time.perf_counter()-started,'quality_released':False,'script_sha256':sha(Path(__file__))}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)); print(json.dumps(receipt))


if __name__=='__main__':
    try: main()
    except Exception as e:
        out=ROOT/sys.argv[1]
        if out.exists():
            with (out/('replay_failure.json' if '--replay' in sys.argv else 'failure.json')).open('x') as f: json.dump({'error':repr(e)},f)
        raise
