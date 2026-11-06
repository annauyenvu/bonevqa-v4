"""Fixed Q/K/V interventions and a nonconstant linear probe; no optimization."""
import json
import math
from pathlib import Path
import torch
from experiments.pf_slake_integrated_v1 import PFRunner
from experiments.pf_masked_variant_v1 import configure_gat
from experiments.lapa_integrated_gate_v1 import REPORTS, batch_data, sha

OUT=REPORTS/'pf_qkv_diagnostic_v1'


def stats(x):
    center=x-x.mean(-2,keepdim=True)
    return {'rms':float(x.square().mean().sqrt()),
            'token_centered_rms':float(center.square().mean().sqrt())}


def main():
    OUT.mkdir(exist_ok=False); torch.set_num_threads(8); torch.manual_seed(42)
    model=PFRunner().eval()
    checkpoint=REPORTS/'pf_slake_integrated_v1/technical_checkpoint.pt'
    saved=torch.load(checkpoint,map_location='cpu');model.load_state_dict(saved['state'],strict=True);del saved
    capture={}; hooks=[]
    for i,layer in enumerate(model.reference.prompt.LF):
        hooks.append(layer.register_forward_hook(lambda m,a,o,i=i:capture.update({f'LF{i+1}':o.detach()})))
    batch,_=batch_data()
    with torch.no_grad():model(batch['images'],batch['questions'])
    for hook in hooks:hook.remove()
    pf=model.slake_pf.double().eval();configure_gat(pf.pf.GAN_layer,'masked_neighbors')
    with torch.no_grad():
        words=model.text.model.get_input_embeddings()(pf.node_ids).double();mask=pf.node_mask.unsqueeze(-1)
        nodes=((words*mask).sum(1)/mask.sum(1)).unsqueeze(0).expand(2,-1,-1)
        real=pf.pf.GAN_layer(nodes,pf.adj_real);rewired=pf.pf.GAN_layer(nodes,pf.adj_rewired)
    prompts=capture['LF6'].double();att=pf.pf.cross_attn_kw.cross_attn
    def project(i,x):return att.linears[i](x).view(2,-1,att.h,att.d_k).transpose(1,2)
    with torch.no_grad():q=project(0,real);qr=project(0,rewired);k=project(1,prompts);v=project(2,prompts)
    gen=torch.Generator().manual_seed(20261026)
    # Diagnostic positive controls only: randomized directions at the SAME RMS.
    def diverse(x):
        z=torch.randn(x.shape,generator=gen,dtype=x.dtype)
        z=z-z.mean(-2,keepdim=True)
        return z/z.square().mean().sqrt()*x.square().mean().sqrt()
    kd,vd=diverse(k),diverse(v)
    probe=torch.randn((2,409,768),generator=gen,dtype=torch.float64)
    probe=probe/probe.norm()
    def forward(query,keys,values):
        probs=(query@keys.transpose(-1,-2)/math.sqrt(att.d_k)).softmax(-1)
        context=(probs@values).transpose(1,2).contiguous().view(2,409,768)
        return pf.pf.cross_attn_kw.layer_norm(att.linears[3](context)),probs
    cases={'actual':(k,v),'constant_keys':(k.mean(-2,keepdim=True).expand_as(k),v),
           'constant_values':(k,v.mean(-2,keepdim=True).expand_as(v)),
           'diverse_keys':(kd,v),'diverse_values':(k,vd),'diverse_both':(kd,vd)}
    records={}
    for name,(keys,values) in cases.items():
        query=q.clone().requires_grad_(True)
        out,probs=forward(query,keys,values)
        with torch.no_grad():other,_=forward(qr,keys,values)
        gradient=torch.autograd.grad((out*probe).sum(),query)[0]
        assert torch.isfinite(gradient).all()
        records[name]={'rewiring_output_max':float((out-other).abs().max()),
                       'query_gradient_max_linear_probe':float(gradient.abs().max()),
                       'attention_deviation_uniform':float((probs-1/32).abs().max())}
    with torch.no_grad():
        manual,_=forward(q,k,v)
        exact=pf.pf.cross_attn_kw(real,prompts,prompts)
        assert torch.allclose(manual,exact,atol=1e-13,rtol=1e-12)
    result={'checkpoint_sha256':sha(checkpoint),'LF_stats':{n:stats(x.double()) for n,x in capture.items()},
            'projected_q':stats(q),'projected_k':stats(k),'projected_v':stats(v),'cases':records,
            'scope':'masked-neighbor diagnostic on existing technical checkpoint; random K/V are positive controls, not medical inputs or a proposed trained configuration',
            'quality_released':False,'source_sha256':sha(Path(__file__))}
    (OUT/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
