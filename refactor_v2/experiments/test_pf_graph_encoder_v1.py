"""Numerical fixtures for graph-off, node-order equivariance and gradients."""
import hashlib,json
from pathlib import Path
import torch
from experiments.pf_graph_encoder_v1 import GraphAttentionEncoder,adjacency_from_edges

ROOT=Path(__file__).resolve().parents[2];O=ROOT/'refactor_v2/reports/pf_slake_graph_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(1);torch.manual_seed(20261028)
    model=GraphAttentionEncoder(16,4);x=torch.randn(6,16,requires_grad=True)
    self_only=adjacency_from_edges(6,[[i,i] for i in range(6)])
    off=model(x,self_only);assert torch.equal(off,torch.nn.functional.elu(model.projection(x)))
    adjacency=self_only.clone();adjacency[0,1]=True;adjacency[1,0]=True
    real=model(x,adjacency);assert not torch.equal(real[0],off[0]);assert torch.equal(real[2:],off[2:])
    order=torch.tensor([5,2,0,4,1,3]);permuted=model(x[order],adjacency[order][:,order])
    assert torch.allclose(permuted,real[order],atol=1e-6)
    loss=real.square().mean();loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().max()>0
    assert model.projection.weight.grad.abs().max()>0
    assert model.attention_neighbor.grad is not None and model.attention_neighbor.grad.abs().max()>0
    # Exercise every prepared409-node adjacency with synthetic features only.
    nodes=read(O/'nodes.json');full=GraphAttentionEncoder(16,8);features=torch.randn(len(nodes),16)
    for name in ('real_graph.json','graph_off.json','degree_placebo.json'):
        result=full(features,adjacency_from_edges(len(nodes),read(O/name)['message_edges']))
        assert result.shape==features.shape and torch.isfinite(result).all()
    evidence={'passed':True,'graph_off_exact_self_projection':True,'isolated_nodes_unaffected':True,'node_permutation_equivariance':True,'input_projection_attention_gradients_nonzero':True,'prepared_graphs_finite':True,'clinical_data_used':False,'model_integration_or_training_released':False,'source_sha256':sha(Path(__file__)),'encoder_source_sha256':sha(ROOT/'refactor_v2/experiments/pf_graph_encoder_v1.py')}
    with (O/'encoder_fixture.json').open('x') as f:json.dump(evidence,f,indent=2)
    print(json.dumps(evidence))
if __name__=='__main__':main()
