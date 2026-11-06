"""Matched read-only graph interventions plus gradients; no quality training."""
import copy
import json
from pathlib import Path
import torch
from experiments.pf_slake_integrated_v1 import PFRunner
from experiments.pf_masked_variant_v1 import configure_gat
from experiments.lapa_integrated_gate_v1 import REPORTS, batch_data, digest, sha

OUT = REPORTS/'pf_masked_variant_v1'


def main():
    OUT.mkdir(exist_ok=False); torch.set_num_threads(8); torch.manual_seed(42)
    model = PFRunner().eval()
    checkpoint = REPORTS/'pf_slake_integrated_v1/technical_checkpoint.pt'
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state['state'], strict=True); del state
    captured = {}
    handle = model.reference.register_forward_hook(lambda m,a,o: captured.update(prompts=o['lf_prompts'].detach()))
    batch,_ = batch_data()
    with torch.no_grad(): model(batch['images'],batch['questions'])
    handle.remove()
    base = model.slake_pf.double().eval()
    with torch.no_grad():
        words = model.text.model.get_input_embeddings()(base.node_ids).double()
        mask = base.node_mask.unsqueeze(-1)
        nodes = ((words*mask).sum(1)/mask.sum(1)).unsqueeze(0).expand(2,-1,-1)
    prompts = captured['prompts'].double()
    records = {}
    reference_digest = digest(base.named_parameters())
    for variant in ('author','mask_only','masked_neighbors'):
        module = copy.deepcopy(base)
        if variant != 'author': configure_gat(module.pf.GAN_layer,variant)
        assert digest(module.named_parameters()) == reference_digest
        outputs = {}
        with torch.no_grad():
            for mode in ('real','repeat','rewired','node_reverse'):
                n = nodes.flip(1) if mode == 'node_reverse' else nodes
                adj = module.adj_rewired if mode == 'rewired' else module.adj_real
                knowledge = module.pf.GAN_layer(n,adj)
                attended = module.pf.cross_attn_kw(knowledge,prompts,prompts)
                summary = module.combined_pool(torch.cat([prompts,module.graph_pool(attended)],1)).squeeze(1).tanh()
                outputs[mode] = {'knowledge':knowledge,'attended':attended,'summary':summary}
        diffs = {mode:{k:float((v-outputs['real'][k]).abs().max()) for k,v in values.items()}
                 for mode,values in outputs.items() if mode!='real'}
        assert all(v==0 for v in diffs['repeat'].values())
        # Isolated self-loop graph must prevent cross-node influence for masked variants.
        with torch.no_grad():
            eye = torch.eye(409,dtype=nodes.dtype)
            perturbed = nodes.clone(); perturbed[:,1,:] += 1
            isolated_change = float((module.pf.GAN_layer(nodes,eye)[:,0]-module.pf.GAN_layer(perturbed,eye)[:,0]).abs().max())
        if variant != 'author': assert isolated_change == 0
        n = nodes.detach().clone().requires_grad_(True)
        out = module.pf(prompts,n,module.adj_real)
        out.square().mean().backward()
        grads = [p.grad for p in module.pf.GAN_layer.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert n.grad is not None and torch.isfinite(n.grad).all()
        records[variant] = {'differences':diffs,'self_loop_other_node_intervention':isolated_change,
                            'gat_gradient_max':max(float(g.abs().max()) for g in grads),
                            'node_gradient_max':float(n.grad.abs().max())}
        del module
    result = {'scope':'float64 matched existing two-update technical weights; actual two TRAIN-image LF features; no optimizer',
              'checkpoint_sha256':sha(checkpoint),'identical_parameter_digest':reference_digest,
              'variants':records,'quality_released':False,
              'sources':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('pf_masked_variant_v1.py')]}}
    (OUT/'receipt.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))


if __name__=='__main__': main()
