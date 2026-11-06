"""Small routing fixtures for the unreleased P wrapper; no backbone loading."""
import hashlib,json
from pathlib import Path
import torch
from src.models.latent_prompt import LatentPromptGenerator
from experiments.pcontext_model_v1 import RoutedContextGenerator

ROOT=Path(__file__).resolve().parents[2];O=ROOT/'refactor_v2/reports/pcontext_component_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(1);torch.manual_seed(20261029)
    m=RoutedContextGenerator(16,4,4,0).eval();m.set_answer_bank(torch.randn(7,16))
    x=torch.randn(4,9,16);mask=torch.ones(4,9,dtype=torch.bool)
    m.runtime_mode='real';real,_=m(x,mask)
    m.runtime_force_zero=True;off,_=m(x,mask)
    inherited,_=LatentPromptGenerator.forward(m,x,mask);assert torch.equal(off,inherited) and not torch.equal(real,off)
    m.runtime_force_zero=False;m.runtime_mode='unconditioned';a,_=m(x,mask);b,_=m(x+2,mask);assert torch.equal(a,b)
    m.runtime_mode='shuffled';m.runtime_group_ids=['a','b','c','d'];shuffled,_=m(x,mask);assert not torch.equal(shuffled,real)
    evidence={'small_generator_routing_passed':True,'forced_zero_exact_inherited_GM':True,'unconditioned_invariant':True,'TRAIN_component_shuffle_intervention':True,'full_backbone_model_gates_pending':True,'model_integration_quality_released':False,'wrapper_source_sha256':sha(ROOT/'refactor_v2/experiments/pcontext_model_v1.py'),'fixture_source_sha256':sha(Path(__file__))}
    with (O/'routing_fixture.json').open('x') as f:json.dump(evidence,f,indent=2)
    print(json.dumps(evidence))
if __name__=='__main__':main()
