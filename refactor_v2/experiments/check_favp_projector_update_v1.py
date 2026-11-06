"""One real TRAIN projector update through frozen Vicuna, with fresh-process replay.

Technical resource-adapted gate, not FAVP quality training or stage-handoff proof.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import torch
from experiments.favp_vicuna_loader_v3 import load_vicuna
from experiments.favp_visual_bridge_v2 import ROOT
from experiments.lapa_integrated_gate_v1 import batch_data
from experiments.prepare_lapa_pilot_v1 import sha

OUT = ROOT / 'reports/favp_projector_update_v1'


def inputs(model, tokenizer, projection, query, question, answer):
    encode = lambda x: tokenizer(x, add_special_tokens=False, return_tensors='pt').input_ids
    # The source caption processor lowercases question text. Fix prospectively here.
    instruction = ('###Human: <Img><ImageHere></Img> [vqa] Based on the image, '
                   'respond to this question with a short answer: ' + question.lower() + '  ###Assistant: ')
    left, right = instruction.split('<ImageHere>')
    embedding = model.get_input_embeddings()
    prefix = torch.cat([embedding(torch.tensor([[tokenizer.bos_token_id]])),
                        embedding(encode(left)), projection(query.float()).half(), embedding(encode(right))], 1)
    answer_ids = encode(answer + '###')
    sequence = torch.cat([prefix, embedding(answer_ids)], 1)
    targets = torch.full(sequence.shape[:2], -100, dtype=torch.long)
    targets[:, prefix.shape[1]:] = answer_ids
    return dict(inputs_embeds=sequence, attention_mask=torch.ones(sequence.shape[:2], dtype=torch.long),
                labels=targets, use_cache=False, return_dict=True)


def run(replay=False):
    torch.set_num_threads(8)
    torch.manual_seed(42)
    start = time.perf_counter()
    if not replay:
        OUT.mkdir(exist_ok=False)
    feature = ROOT / 'reports/favp_visual_bridge_v2/features.pt'
    saved = torch.load(feature, map_location='cpu', weights_only=True)
    batch, answers = batch_data()
    assert batch['image_ids'][0] == saved['image_id']
    model, tokenizer, audit = load_vicuna()
    projection = torch.nn.Linear(768, 4096).float()
    source_paths = [Path(__file__), Path(__file__).with_name('favp_vicuna_loader_v3.py')]
    if replay:
        state = torch.load(OUT / 'projector.pt', map_location='cpu', weights_only=True)
        assert state['feature_sha256'] == sha(feature)
        assert state['sources'] == {str(p): sha(p) for p in source_paths}
        projection.load_state_dict(state['projection'], strict=True)
        model.eval()
        with torch.no_grad():
            result = model(**inputs(model, tokenizer, projection, saved['queries'], batch['questions'][0], answers[0]))
        assert torch.equal(result.logits, state['logits'])
        assert torch.equal(result.loss, state['loss'])
        receipt = {'exact_logits': True, 'exact_loss': True, 'strict_projector_reload': True,
                   'checkpoint_sha256': sha(OUT / 'projector.pt'), 'seconds': time.perf_counter()-start}
        (OUT / 'replay.json').write_text(json.dumps(receipt, indent=2))
        return
    initial = {k: v.detach().clone() for k, v in projection.state_dict().items()}
    # Frozen parameters cannot update; record version counters to detect in-place mutation.
    versions = {k: p._version for k, p in model.named_parameters()}
    assert all(not p.requires_grad for p in model.parameters())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    optimizer = torch.optim.SGD(projection.parameters(), lr=1e-3)
    tick = time.perf_counter()
    result = model(**inputs(model, tokenizer, projection, saved['queries'], batch['questions'][0], answers[0]))
    assert torch.isfinite(result.loss)
    before = float(result.loss.detach())
    result.loss.backward()
    gradients = {k: float(p.grad.float().norm()) for k, p in projection.named_parameters()}
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0 for p in projection.parameters())
    assert all(p.grad is None and p._version == versions[k] for k, p in model.named_parameters())
    torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    changed = {k: not torch.equal(v, initial[k]) for k, v in projection.state_dict().items()}
    assert all(changed.values())
    update_seconds = time.perf_counter()-tick
    del result
    model.gradient_checkpointing_disable()
    model.eval()
    with torch.no_grad():
        after = model(**inputs(model, tokenizer, projection, saved['queries'], batch['questions'][0], answers[0]))
    assert torch.isfinite(after.loss) and torch.isfinite(after.logits).all()
    assert all(p._version == versions[k] for k, p in model.named_parameters())
    torch.save({'projection': projection.state_dict(), 'logits': after.logits, 'loss': after.loss,
                'feature_sha256': sha(feature), 'sources': {str(p): sha(p) for p in source_paths}}, OUT / 'projector.pt')
    receipt = {'image_id': saved['image_id'], 'updates': 1, 'gradient_norms': gradients,
               'projector_changed': changed, 'frozen_parameter_versions_unchanged': True,
               'frozen_gradients_absent': True, 'loss_before': before, 'loss_after': float(after.loss),
               'update_seconds': update_seconds, 'total_seconds': time.perf_counter()-start,
               'peak_rss_raw': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'audit': audit,
               'quality_released': False, 'scope': 'cached actual image queries; random FP32 projector only; frozen native Vicuna; checkpointed CPU FP16 technical gradient gate, not full FAVP training',
               'question_policy': 'lowercase prospectively; no assessment tuning',
               'stage_handoff': 'unverified; no pretrained Vicuna projection claim'}
    (OUT / 'update.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    try:
        run(args.replay)
    except Exception as exc:
        if OUT.exists():
            with (OUT / ('replay_failure.json' if args.replay else 'failure.json')).open('x') as f:
                json.dump({'error': repr(exc)}, f)
        raise
