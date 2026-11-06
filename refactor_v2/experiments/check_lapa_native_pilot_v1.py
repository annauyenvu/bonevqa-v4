"""First real native-backbone optimizer/reload probe, not quality training."""
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.lapa_native_pilot_v1 import NativePilot
from experiments.lapa_integrated_gate_v1 import REPORTS,batch_data,digest
from experiments.lapa_reference_endpoint_v1 import reference_loss
from experiments.prepare_lapa_pilot_v1 import sha

OUT=REPORTS/'lapa_native_pilot_probe_v1'


def main():
    replay='--replay' in sys.argv
    if not replay:OUT.mkdir(exist_ok=False)
    torch.set_num_threads(8);start=time.perf_counter()
    m=NativePilot('pf_author').eval();batch,answers=batch_data()
    names={n for n,p in m.named_parameters(remove_duplicate=False) if p.requires_grad}
    fixed=digest((n,t) for n,t in m.state_dict().items() if n not in names)
    if replay:
        saved=torch.load(OUT/'trainable.pt',map_location='cpu',weights_only=True)
        assert fixed==saved['fixed_digest']
        state=m.state_dict();state.update(saved['trainable']);m.load_state_dict(state,strict=True)
        assert digest(m.state_dict().items())==saved['final_digest']
        with torch.no_grad():logits,_=m(batch['images'],batch['questions'])
        assert torch.equal(logits,saved['logits'])
        (OUT/'replay.json').write_text(json.dumps({'full_state_exact':True,'logits_exact':True}));return
    tokenizer=m.backbone.tokenizer
    length=max(len(tokenizer.encode(a,add_special_tokens=True)) for a in m.vocab)
    ids=tokenizer(answers,padding='max_length',max_length=length,return_tensors='pt')['input_ids']
    with torch.no_grad():words=m.backbone.text.get_input_embeddings()(ids)
    targets=F.one_hot(torch.tensor([m.vocab.index(a) for a in answers]),41).float()
    groups={'MM':list(m.backbone.reference.text_layers.parameters()),'GM':list(m.backbone.reference.prompt.generation.parameters()),
            'LF':list(m.backbone.reference.prompt.LF.parameters()),'GAT':list(m.slake_pf.pf.GAN_layer.parameters()),'head':list(m.head.parameters())}
    before={k:[p.detach().clone() for p in values] for k,values in groups.items()}
    opt=m.optimizer();m.train();steps=[]
    for step in range(2):
        tick=time.perf_counter();opt.zero_grad();logits,g=m(batch['images'],batch['questions'])
        loss=reference_loss(logits,targets,g,words,.1)['total'];assert torch.isfinite(loss);loss.backward()
        gradients={k:max(float(p.grad.abs().max()) for p in values if p.grad is not None) for k,values in groups.items()}
        assert all(x>0 for x in gradients.values())
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);opt.step()
        steps.append({'seconds':time.perf_counter()-tick,'loss':float(loss),'gradients':gradients})
    changed={k:any(not torch.equal(a,p) for a,p in zip(before[k],values)) for k,values in groups.items()};assert all(changed.values())
    assert fixed==digest((n,t) for n,t in m.state_dict().items() if n not in names)
    m.eval()
    with torch.no_grad():logits,_=m(batch['images'],batch['questions'])
    torch.save({'trainable':{n:t for n,t in m.state_dict().items() if n in names},'fixed_digest':fixed,
                'final_digest':digest(m.state_dict().items()),'logits':logits},OUT/'trainable.pt')
    del opt,m,before,groups;gc.collect()
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--replay'],check=True)
    result={'arm':'pf_author','scope':'two actual TRAIN records, frozen native M3AE image/text encoders, pretrained MM plus new prompts/head; graph409 adaptation',
            'updates':2,'steps':steps,'updated_groups':changed,'fixed_state_unchanged':True,
            'replay':json.loads((OUT/'replay.json').read_text()),'seconds':time.perf_counter()-start,
            'quality_released':False,'sources':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('lapa_native_pilot_v1.py')]}}
    (OUT/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():
            with (OUT/('replay_failure.json' if '--replay' in sys.argv else 'failure.json')).open('x') as f:json.dump({'error':repr(e)},f)
        raise
