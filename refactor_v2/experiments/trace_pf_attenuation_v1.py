"""Read-only numerical trace of author PF on the existing technical checkpoint."""
import json
import math
import torch
from torch.nn import functional as F
from experiments.pf_slake_integrated_v1 import PFRunner
from experiments.lapa_integrated_gate_v1 import REPORTS, batch_data, sha
from pathlib import Path

OUT=REPORTS/'pf_attenuation_trace_v1'


def spread(x):
    return float((x-x.mean(1,keepdim=True)).abs().max())


def layer_trace(layer,x,adj):
    wh=x@layer.W
    e=layer.leakyrelu(wh@layer.a[:layer.out_features]+(wh@layer.a[layer.out_features:]).transpose(1,2))
    weights=torch.where(adj>0,e,torch.zeros_like(e)).softmax(1)
    manual=weights@wh
    if layer.concat: manual=F.elu(manual)
    actual=layer(x,adj)
    assert torch.allclose(actual,manual,atol=1e-14,rtol=1e-12)
    # Author uses column normalization; column nonedge mass is measured as such.
    nonedge=(weights*(adj==0)).sum(1).mean()
    return actual, {'input_node_spread':spread(x),'output_node_spread':spread(actual),
                    'edge_logit_max':float(e[adj.bool().unsqueeze(0).expand_as(e)].abs().max()),
                    'mean_column_nonedge_mass':float(nonedge),
                    'max_weight_deviation_uniform':float((weights-1/x.shape[1]).abs().max())}


def main():
    OUT.mkdir(exist_ok=False); torch.set_num_threads(8); torch.manual_seed(42)
    model=PFRunner().eval()
    checkpoint=REPORTS/'pf_slake_integrated_v1/technical_checkpoint.pt'
    saved=torch.load(checkpoint,map_location='cpu'); model.load_state_dict(saved['state'],strict=True); del saved
    captured={}
    hook=model.reference.register_forward_hook(lambda m,a,o: captured.update(prompts=o['lf_prompts'].detach()))
    batch,_=batch_data()
    with torch.no_grad(): model(batch['images'],batch['questions'])
    hook.remove()
    pf=model.slake_pf.double().eval()
    with torch.no_grad():
        words=model.text.model.get_input_embeddings()(pf.node_ids).double()
        mask=pf.node_mask.unsqueeze(-1)
        nodes=((words*mask).sum(1)/mask.sum(1)).unsqueeze(0).expand(2,-1,-1)
        prompts=captured['prompts'].double()
        first,s1=layer_trace(pf.pf.GAN_layer.attentions[0],nodes,pf.adj_real)
        knowledge,s2=layer_trace(pf.pf.GAN_layer.out_att,first,pf.adj_real)
        att=pf.pf.cross_attn_kw.cross_attn
        def project(i,x):
            return att.linears[i](x).view(2,-1,att.h,att.d_k).transpose(1,2)
        q,k,v=project(0,knowledge),project(1,prompts),project(2,prompts)
        scores=q@k.transpose(-1,-2)/math.sqrt(att.d_k)
        probs=scores.softmax(-1)
        actual=pf.pf.cross_attn_kw(knowledge,prompts,prompts)
        uniform_context=v.mean(2,keepdim=True).expand(-1,-1,409,-1).transpose(1,2).contiguous().view(2,409,768)
        uniform_out=pf.pf.cross_attn_kw.layer_norm(att.linears[3](uniform_context))
        zeros_out=pf.pf.cross_attn_kw(torch.zeros_like(knowledge),prompts,prompts)
        cross={'score_range_across_prompt_keys':float((scores.max(-1).values-scores.min(-1).values).max()),
               'max_probability_deviation_uniform':float((probs-1/32).abs().max()),
               'value_token_spread':float((v-v.mean(2,keepdim=True)).abs().max()),
               'actual_minus_uniform_attention_max':float((actual-uniform_out).abs().max()),
               'actual_minus_zero_graph_query_max':float((actual-zeros_out).abs().max())}
    # Absolute derivatives quantify the graph route separately from its shared attention parameters.
    knowledge=knowledge.detach().requires_grad_(True)
    summary=pf.combined_pool(torch.cat([prompts,pf.graph_pool(pf.pf.cross_attn_kw(knowledge,prompts,prompts))],1)).squeeze(1).tanh()
    derivative=torch.autograd.grad(summary.square().sum(),knowledge)[0]
    receipt={'checkpoint_sha256':sha(checkpoint),'GAT_layer1':s1,'GAT_layer2':s2,'PF_attention':cross,
             'summary_squared_norm_gradient_to_graph_max':float(derivative.abs().max()),
             'scope':'float64 eval, existing two-update technical checkpoint, no new optimizer or quality data',
             'source_sha256':sha(Path(__file__)),'quality_released':False}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)); print(json.dumps(receipt))


if __name__=='__main__': main()
