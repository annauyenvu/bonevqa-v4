"""Real-image regression gate for explicit v2 graph variant, no quality run."""
import json
import subprocess
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.pf_reconstructed_v2 import ReconstructedPFRunner
from experiments.pf_slake_integrated_v1 import PFRunner
from experiments.lapa_integrated_gate_v1 import REPORTS, batch_data, digest, sha
from experiments.lapa_reference_endpoint_v1 import reference_loss

OUT=REPORTS/'pf_reconstructed_v2'


def diversity(x):
    return float((x-x.mean(1,keepdim=True)).square().mean().sqrt())


def trace(model,batch):
    values={'latent':diversity(model.reference.prompt.latent.unsqueeze(0))}; hooks=[]
    def record(name):
        return lambda m,a,o: values.update({name:diversity(o.detach())})
    hooks.append(model.reference.prompt.generation.register_forward_hook(record('GM')))
    for i,block in enumerate(model.reference.prompt.LF):
        for name in ('cross_attn_text','cross_attn_image','cross_attn_img_text','doc_ffd'):
            hooks.append(getattr(block,name).register_forward_hook(record(f'LF{i+1}.{name}')))
    with torch.no_grad(): model(batch['images'],batch['questions'])
    for hook in hooks: hook.remove()
    return values


def outputs(model,batch):
    result={}
    with torch.no_grad():
        for mode in ('real','rewired','bypass'):
            model.pf_mode=mode
            result[mode]=model(batch['images'],batch['questions'])[0]
    model.pf_mode='real'
    return result


def main():
    torch.set_num_threads(8);torch.manual_seed(42);start=time.perf_counter()
    replay='--replay' in sys.argv
    if not replay: OUT.mkdir(exist_ok=False)
    model=ReconstructedPFRunner().eval();batch,answers=batch_data()
    if replay:
        saved=torch.load(OUT/'technical_checkpoint.pt',map_location='cpu')
        model.load_state_dict(saved['state'],strict=True)
        actual=outputs(model,batch)
        checks={k:torch.equal(v,saved['outputs'][k]) for k,v in actual.items()}
        assert all(checks.values())
        (OUT/'replay.json').write_text(json.dumps(checks));return
    binding=json.loads((REPORTS/'lapa_port_backbone_binding_v1/cache_binding.json').read_text())
    for entry in binding['models']:
        for file in entry['files']:assert sha(Path(file['resolved_path']))==file['sha256']
    pre=REPORTS/'step4_localization_v3_1/B_PREtask.pt'
    release=json.loads((REPORTS/'vnext_local_tokens_v1/training_release.json').read_text())
    assert sha(pre)==release['dependencies'][str(pre)]
    state=torch.load(pre,map_location='cpu')['state'];visual={k[7:]:v for k,v in state.items() if k.startswith('visual.')}
    assert not model.visual.load_state_dict(visual,strict=False).unexpected_keys
    assert all(torch.equal(v,model.visual.state_dict()[k]) for k,v in visual.items());del state,visual
    initial_digest=digest(model.named_parameters())
    # Trace author construction BEFORE and AFTER its historical disposable updates.
    original=PFRunner().eval();original.load_state_dict(model.state_dict(),strict=True)
    pre_trace=trace(original,batch)
    saved=torch.load(REPORTS/'pf_slake_integrated_v1/technical_checkpoint.pt',map_location='cpu')
    original.load_state_dict(saved['state'],strict=True);del saved
    post_trace=trace(original,batch);del original
    before_outputs=outputs(model,batch)
    changes={mode:float((before_outputs['real']-v).abs().max()) for mode,v in before_outputs.items() if mode!='real'}
    assert all(v>1e-5 for v in changes.values()),changes
    # Query residual must preserve graph differences even when all prompt slots match.
    pf=model.slake_pf
    with torch.no_grad():
        embedding=model.text.model.get_input_embeddings()
        words=embedding(pf.node_ids);mask=pf.node_mask.unsqueeze(-1)
        nodes=((words*mask).sum(1)/mask.sum(1)).unsqueeze(0).expand(2,-1,-1)
        constant=torch.ones(2,32,768)
        real=pf.pf(constant,nodes,pf.adj_real);rewired=pf.pf(constant,nodes,pf.adj_rewired)
        constant_diff=float((real-rewired).abs().max());assert constant_diff>1e-5
        eye=torch.eye(409);other=nodes.clone();other[:,1]+=1
        isolation=float((pf.pf.GAN_layer(nodes,eye)[:,0]-pf.pf.GAN_layer(other,eye)[:,0]).abs().max())
        assert isolation==0
    length=max(len(model.text.tokenizer.encode(a,add_special_tokens=True)) for a in model.vocab)
    ids=model.text.tokenizer(answers,padding='max_length',max_length=length,return_tensors='pt')['input_ids']
    with torch.no_grad():words=model.text.model.get_input_embeddings()(ids)
    targets=F.one_hot(torch.tensor([model.vocab.index(a) for a in answers]),len(model.vocab)).float()
    frozen=digest((n,p) for n,p in model.named_parameters() if not p.requires_grad)
    groups={'GAT':list(model.slake_pf.pf.GAN_layer.named_parameters()),'PF_attention':list(model.slake_pf.pf.cross_attn_kw.named_parameters()),
            'graph_pool':list(model.slake_pf.graph_pool.named_parameters()),'head':list(model.head.named_parameters())}
    initial={k:digest(v) for k,v in groups.items()};optimizer=model.optimizer();steps=[];model.train()
    for step in range(2):
        optimizer.zero_grad();logits,generated=model(batch['images'],batch['questions'])
        losses=reference_loss(logits,targets,generated,words,.1)
        gat=[p for _,p in groups['GAT']]
        answer_grads=torch.autograd.grad(losses['answer'],gat,retain_graph=True,allow_unused=True)
        assert any(g is not None and g.abs().max()>0 for g in answer_grads)
        cs_grads=torch.autograd.grad(losses['consistency_sum'],gat,retain_graph=True,allow_unused=True)
        assert all(g is None for g in cs_grads)
        losses['total'].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        steps.append({'loss':float(losses['total']),'answer_to_GAT_gradient_max':max(float(g.abs().max()) for g in answer_grads if g is not None)})
    changed={k:initial[k]!=digest(v) for k,v in groups.items()};assert all(changed.values())
    assert frozen==digest((n,p) for n,p in model.named_parameters() if not p.requires_grad)
    model.eval();final=outputs(model,batch)
    torch.save({'state':model.state_dict(),'outputs':final},OUT/'technical_checkpoint.pt')
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--replay'],check=True)
    result={'scope':'new algorithm variant, fresh PRE-update initialization, two TRAIN records/two disposable updates only',
            'initial_digest':initial_digest,'author_prompt_diversity_before_updates':pre_trace,'author_prompt_diversity_after_two_updates':post_trace,
            'initial_logit_interventions':changes,'constant_prompt_graph_difference':constant_diff,'self_only_isolation':isolation,
            'steps':steps,'changed_groups':changed,'frozen_unchanged':True,'fresh_replay':json.loads((OUT/'replay.json').read_text()),
            'seconds':time.perf_counter()-start,'quality_released':False,
            'sources':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('pf_reconstructed_v2.py'),Path(__file__).with_name('pf_masked_variant_v1.py')]}}
    (OUT/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():
            with (OUT/('replay_failure.json' if '--replay' in sys.argv else 'failure.json')).open('x') as f:json.dump({'error':repr(error)},f)
        raise
