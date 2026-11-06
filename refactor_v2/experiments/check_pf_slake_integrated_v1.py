"""Bounded real-image technical gates; never starts quality training."""
import json
import subprocess
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.pf_slake_integrated_v1 import PFRunner
from experiments.lapa_integrated_gate_v1 import batch_data, digest, REPORTS, sha
from experiments.lapa_reference_endpoint_v1 import reference_loss

OUT = REPORTS / 'pf_slake_integrated_v1'


def main():
    torch.set_num_threads(8); torch.manual_seed(42)
    replay = '--replay' in sys.argv
    if not replay: OUT.mkdir(exist_ok=False)
    start = time.perf_counter()
    binding = json.loads((REPORTS/'lapa_port_backbone_binding_v1/cache_binding.json').read_text())
    for model in binding['models']:
        for file in model['files']:
            assert sha(Path(file['resolved_path'])) == file['sha256']
    runner = PFRunner().eval()
    batch, answers = batch_data()
    if replay:
        saved = torch.load(OUT/'technical_checkpoint.pt', map_location='cpu')
        runner.load_state_dict(saved['state'], strict=True)
        checks = {}
        with torch.no_grad():
            for mode in ('real','rewired','bypass'):
                runner.pf_mode = mode
                checks[mode] = torch.equal(runner(batch['images'],batch['questions'])[0],saved['outputs'][mode])
        assert all(checks.values())
        (OUT/'replay.json').write_text(json.dumps(checks))
        return
    source = REPORTS/'step4_localization_v3_1/B_PREtask.pt'
    release = json.loads((REPORTS/'vnext_local_tokens_v1/training_release.json').read_text())
    assert sha(source) == release['dependencies'][str(source)]
    state = torch.load(source,map_location='cpu')['state']
    visual = {k[7:]:v for k,v in state.items() if k.startswith('visual.')}
    assert not runner.visual.load_state_dict(visual,strict=False).unexpected_keys
    assert all(torch.equal(v,runner.visual.state_dict()[k]) for k,v in visual.items())
    del state, visual
    initial_digest = digest(runner.named_parameters())
    # All modes are switches on the SAME pre-update tensors: exact matched init.
    with torch.no_grad():
        initial_outputs = {}
        for mode in ('real','rewired','bypass'):
            runner.pf_mode = mode
            initial_outputs[mode] = runner(batch['images'],batch['questions'])[0]
        runner.pf_mode = 'real'
        original_ids = runner.slake_pf.node_ids.clone()
        original_mask = runner.slake_pf.node_mask.clone()
        runner.slake_pf.node_ids.copy_(original_ids.flip(0))
        runner.slake_pf.node_mask.copy_(original_mask.flip(0))
        intervened = runner(batch['images'],batch['questions'])[0]
        runner.slake_pf.node_ids.copy_(original_ids)
        runner.slake_pf.node_mask.copy_(original_mask)
    differences = {m: float((initial_outputs['real']-v).abs().max()) for m,v in initial_outputs.items() if m!='real'}
    differences['node_order_intervention'] = float((initial_outputs['real']-intervened).abs().max())
    assert all(v > 0 for v in differences.values()), differences
    length = max(len(runner.text.tokenizer.encode(a,add_special_tokens=True)) for a in runner.vocab)
    ids = runner.text.tokenizer(answers,padding='max_length',max_length=length,truncation=False,return_tensors='pt')['input_ids']
    with torch.no_grad(): words = runner.text.model.get_input_embeddings()(ids)
    targets = F.one_hot(torch.tensor([runner.vocab.index(a) for a in answers]),len(runner.vocab)).float()
    groups = {'PF': list(runner.slake_pf.pf.named_parameters()),
              'graph_pool': list(runner.slake_pf.graph_pool.named_parameters()),
              'combined_pool': list(runner.slake_pf.combined_pool.named_parameters()),
              'LF': list(runner.reference.prompt.LF.named_parameters()),
              'GM': list(runner.reference.prompt.generation.named_parameters()),
              'head': list(runner.head.named_parameters()),
              'visual_lora': [(n,p) for n,p in runner.visual.named_parameters() if 'lora_' in n]}
    before = {k:digest(v) for k,v in groups.items()}
    frozen = digest((n,p) for n,p in runner.named_parameters() if not p.requires_grad)
    optimizer = runner.optimizer(); steps=[]
    runner.train()
    for step in range(2):
        optimizer.zero_grad(); tick=time.perf_counter()
        logits, generated = runner(batch['images'],batch['questions'])
        losses = reference_loss(logits, targets, generated, words, .1)
        if step == 0:
            pf_params = [p for _,p in groups['PF']]
            route = torch.autograd.grad(losses['consistency_sum'],pf_params,allow_unused=True,retain_graph=True)
            assert all(g is None for g in route)
            answer_route = torch.autograd.grad(losses['answer'],pf_params,allow_unused=True,retain_graph=True)
            assert any(g is not None and g.abs().max()>0 for g in answer_route)
        losses['total'].backward()
        gradients = {}
        for name,params in groups.items():
            gs = [p.grad for _,p in params if p.grad is not None]
            assert gs and all(torch.isfinite(g).all() for g in gs),name
            gradients[name]=max(float(g.abs().max()) for g in gs)
            assert gradients[name]>0,name
        torch.nn.utils.clip_grad_norm_(runner.parameters(),1.)
        optimizer.step()
        steps.append({'seconds':time.perf_counter()-tick,'loss':float(losses['total']),'gradient_max':gradients})
    changed = {k:before[k]!=digest(v) for k,v in groups.items()}
    assert all(changed.values())
    assert frozen == digest((n,p) for n,p in runner.named_parameters() if not p.requires_grad)
    runner.eval(); outputs={}
    with torch.no_grad():
        for mode in ('real','rewired','bypass'):
            runner.pf_mode=mode
            outputs[mode]=runner(batch['images'],batch['questions'])[0]
    torch.save({'state':runner.state_dict(),'outputs':outputs},OUT/'technical_checkpoint.pt')
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--replay'],check=True)
    receipt={'scope':'Shared BiomedCLIP/PubMedBERT PF with reconstructed SLAKE; two real TRAIN records and two disposable train-mode AdamW updates of real route only',
             'images':batch['image_ids'],'initial_digest':initial_digest,'matched_initial_modes':True,
             'node_tokens':int(runner.slake_pf.node_mask.sum()),'tokenizer':'actual backbone PubMedBERT',
             'intervention_max_logit_differences':differences,'steps':steps,'changed_groups':changed,
             'frozen_unchanged':True,'answer_routes_to_PF':True,'consistency_does_not_route_to_PF':True,
             'fresh_replay':json.loads((OUT/'replay.json').read_text()),'seconds':time.perf_counter()-start,
             'quality_released':False,'source_hashes':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('pf_slake_integrated_v1.py'),Path(__file__).with_name('pf_slake_reference_v1.py')]}}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)); print(json.dumps(receipt))


if __name__ == '__main__':
    try: main()
    except Exception as error:
        if OUT.exists():
            name='replay_failure.json' if '--replay' in sys.argv else 'failure.json'
            with (OUT/name).open('x') as f: json.dump({'error':repr(error)},f)
        raise
