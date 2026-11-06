"""Float64 PF stage interventions on real-image LF features, no updates."""
import json
from pathlib import Path
import torch
from experiments.pf_slake_integrated_v1 import PFRunner
from experiments.lapa_integrated_gate_v1 import batch_data, REPORTS

OUT = REPORTS/'pf_slake_sensitivity_v1'


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(8); torch.manual_seed(42)
    model=PFRunner().eval()
    saved=torch.load(REPORTS/'pf_slake_integrated_v1/technical_checkpoint.pt',map_location='cpu')
    model.load_state_dict(saved['state'],strict=True); del saved
    captured={}
    handle=model.reference.register_forward_hook(lambda module,args,out: captured.update(prompts=out['lf_prompts'].detach()))
    batch,_=batch_data()
    with torch.no_grad(): model(batch['images'],batch['questions'])
    handle.remove()
    pf=model.slake_pf.double().eval()
    with torch.no_grad():
        words=model.text.model.get_input_embeddings()(pf.node_ids).double()
        mask=pf.node_mask.unsqueeze(-1)
        nodes=((words*mask).sum(1)/mask.sum(1)).unsqueeze(0).expand(2,-1,-1)
        prompts=captured['prompts'].double()
        results={}
        for mode in ('real','repeat','rewired','node_reverse','bypass','prompt_reverse'):
            n=nodes.flip(1) if mode=='node_reverse' else nodes
            p=prompts.flip(1) if mode=='prompt_reverse' else prompts
            adj=pf.adj_rewired if mode=='rewired' else pf.adj_real
            knowledge=n if mode=='bypass' else pf.pf.GAN_layer(n,adj)
            attended=pf.pf.cross_attn_kw(knowledge,p,p)
            slots=pf.graph_pool(attended)
            summary=pf.combined_pool(torch.cat([p,slots],1)).squeeze(1).tanh()
            results[mode]={'knowledge':knowledge,'attended':attended,'slots':slots,'summary':summary}
        differences={mode:{stage:float((tensor-results['real'][stage]).abs().max()) for stage,tensor in values.items()}
                     for mode,values in results.items() if mode!='real'}
        assert all(v==0 for v in differences['repeat'].values())
        receipt={'precision':'float64 PF arithmetic; actual trained-two-step LF features and backbone word embeddings',
                 'max_abs_differences':differences,
                 'prompt_token_spread':float((prompts-prompts.mean(1,keepdim=True)).abs().max()),
                 'quality_release':False,'no_updates':True}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)); print(json.dumps(receipt))


if __name__=='__main__': main()
