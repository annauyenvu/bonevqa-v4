"""Synthetic semantic-routing and gradient fixtures, no VQA training."""
import hashlib,json
from pathlib import Path
import torch
from experiments.pcontext_component_v1 import ContextualLatentGenerator

ROOT=Path(__file__).resolve().parents[2];O=ROOT/'refactor_v2/reports/pcontext_component_v1'
def main():
    assert not O.exists();O.mkdir();torch.set_num_threads(1);torch.manual_seed(20261029)
    model=ContextualLatentGenerator(16,4,4,0).eval();model.set_answer_bank(torch.randn(7,16))
    x=torch.randn(4,9,16,requires_grad=True);mask=torch.ones(4,9,dtype=torch.bool);mask[:,-2:]=False
    a,_=model(x,mask);changed=x.detach().clone();changed[0]+=1
    b,_=model(changed,mask);assert not torch.equal(a[0],b[0]) and torch.equal(a[1:],b[1:])
    a0,_=model(x,mask,'unconditioned');b0,_=model(changed,mask,'unconditioned');assert torch.equal(a0,b0)
    shuffled,_=model(x,mask,'shuffled',['a','b','c','d']);assert not torch.equal(a,shuffled)
    loss=a.square().mean();loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().max()>0
    assert model.context_gate.grad is not None and model.context_gate.grad.abs()>0
    assert model.context_cross.attn.in_proj_weight.grad is not None if hasattr(model.context_cross,'attn') else True
    grads={k:float(p.grad.abs().max()) for k,p in model.named_parameters() if p.grad is not None and k.startswith('context_')}
    assert any(v>0 for k,v in grads.items() if k.startswith('context_cross.'))
    result={'passed':True,'real_context_intervention_changed_only_target_row':True,'unconditioned_invariant':True,'component_disjoint_shuffle_changes_output':True,'input_and_context_parameter_gradients_nonzero':True,'gradient_maxima':grads,'clinical_data_used':False,'model_integration_or_quality_released':False,'component_source_sha256':hashlib.sha256((ROOT/'refactor_v2/experiments/pcontext_component_v1.py').read_bytes()).hexdigest()}
    with (O/'synthetic_fixture.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
