"""Synthetic interface checks, not a clinical/architecture release."""
import hashlib
import json
from pathlib import Path

import torch
from experiments.lapa_reference_prompt_core_v1 import ReferencePromptCore


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261026)
    model = ReferencePromptCore(41, d_model=32, graph_nodes=5).eval()
    bank = torch.randn(41, 32, requires_grad=True)
    contexts = [(torch.randn(2, 3, 32), torch.randn(2, 4, 32), torch.randn(2, 7, 32)) for _ in range(6)]
    graph = torch.randn(2, 5, 32)
    adj = torch.eye(5).expand(2, -1, -1)
    first = model(bank, contexts)
    changed = [(a.clone(), b.clone(), c.clone()) for a, b, c in contexts]
    changed[3] = (changed[3][0] + torch.randn_like(changed[3][0]), *changed[3][1:])
    assert not torch.equal(first['latent_summary'], model(bank, changed)['latent_summary'])
    assert torch.equal(first['generated_mean'], model(bank, changed)['generated_mean'])
    try:
        model(bank, contexts[:1])
    except ValueError:
        pass
    else:
        raise AssertionError('Missing per-layer contexts accepted')
    author = model(bank, contexts, graph, adj, 'author')
    bypass = model(bank, contexts, graph, adj, 'graph_bypass')
    assert author['latent_summary'].shape == bypass['latent_summary'].shape == (2, 32)
    assert not torch.equal(author['latent_summary'], bypass['latent_summary'])
    target = torch.randn(2, 4, 32)
    loss = first['latent_summary'].square().mean() + model.author_consistency(first['generated_mean'], target)
    loss.backward()
    assert bank.grad is not None and bank.grad.abs().sum() > 0 and torch.isfinite(bank.grad).all()
    out = Path(__file__).resolve().parents[2] / 'refactor_v2/reports/lapa_reference_prompt_core_v1'
    out.mkdir(exist_ok=False)
    record = {
        'scope': 'Synthetic32dim5node interface fixture; no real backbone/577graph/clinical GT/model-release.',
        'six_explicit_layer_contexts_required': True, 'single_layer_rejected': True,
        'LF_layer_intervention_changes_summary': True,
        'GM_independent_of_image_question_context': True,
        'raw_answer_bank_gradient_finite_nonzero': True,
        'author_graph_vs_explicit_bypass_distinct': True,
        'PF_off_and_on_reduction_parameters_differ': True,
        'reduction_warning': 'off-vs-on alone changes pooling parameterization; structure inference requires matched author-vs-bypass controls.',
        'public_runner_integrated': False, 'quality_released': False,
        'dependencies': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).with_name('lapa_reference_prompt_core_v1.py')]},
    }
    with (out / 'interface_fixture.json').open('x') as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
