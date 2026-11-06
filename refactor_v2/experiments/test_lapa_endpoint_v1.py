"""Endpoint arithmetic/update/strict fresh-process replay on synthetic features."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.lapa_reference_endpoint_v1 import ReferenceAnswerHead, reference_loss

OUT = Path(__file__).resolve().parents[1]/'reports/lapa_endpoint_fixture_v1'
torch.set_num_threads(8)


def main():
    if '--replay' in sys.argv:
        saved = torch.load(OUT/'state.pt', map_location='cpu')
        head = ReferenceAnswerHead(768,41).eval()
        head.load_state_dict(saved['head'], strict=True)
        with torch.no_grad(): actual = head(saved['features'])
        assert torch.equal(actual, saved['expected'])
        (OUT/'replay.json').write_text(json.dumps({'strict_reload':True,'exact_logits':True}))
        return
    OUT.mkdir(exist_ok=False)
    torch.manual_seed(42)
    head = ReferenceAnswerHead(768,41)
    features = torch.randn(2,768)
    generated = torch.randn(2,768,requires_grad=True)
    words = torch.randn(2,8,768)
    targets = F.one_hot(torch.tensor([0,1]),41).float()
    optimizer = torch.optim.SGD(head.parameters(),lr=1e-3)
    before = {k:v.clone() for k,v in head.state_dict().items()}
    losses = []
    for _ in range(2):
        optimizer.zero_grad(); generated.grad = None
        logits = head(features)
        result = reference_loss(logits,targets,generated,words)
        independent = F.binary_cross_entropy_with_logits(logits,targets)*41 + .1*(1-F.cosine_similarity(generated,words.mean(1))).sum()
        assert torch.equal(result['total'], independent)
        answer_to_gm = torch.autograd.grad(result['answer'],generated,allow_unused=True,retain_graph=True)[0]
        assert answer_to_gm is None
        result['total'].backward()
        assert generated.grad is not None and torch.isfinite(generated.grad).all() and generated.grad.abs().max()>0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        optimizer.step(); losses.append(result['total'].item())
    changed = [k for k,v in head.state_dict().items() if not torch.equal(v,before[k])]
    assert len(changed)==len(before)
    head.eval()
    with torch.no_grad(): expected = head(features)
    torch.save({'head':head.state_dict(),'features':features,'expected':expected},OUT/'state.pt')
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--replay'],check=True)
    receipt = {'scope':'Synthetic fixed features; endpoint only, no encoder/full-model gate or performance claim.',
               'updates':2,'losses':losses,'changed_parameter_tensors':changed,
               'source_objective_arithmetic_exact':True,'fresh_process_replay':json.loads((OUT/'replay.json').read_text()),
               'quality_release':False,'dependencies':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('lapa_reference_endpoint_v1.py')]}}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
