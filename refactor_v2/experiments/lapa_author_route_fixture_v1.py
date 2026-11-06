"""Exact pinned author prompt primitives in a synthetic wiring harness, not VQA.

No author backbone/assets/dependencies are substituted or implicitly downloaded.
"""
import ast
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'refactor_v2/references/lapa_author_677faf90'
ATTEMPT = os.environ.get('BONEVQA_LAPA_FIXTURE_ATTEMPT', 'lapa_author_route_fixture_v1')
assert ATTEMPT in ('lapa_author_route_fixture_v1', 'lapa_author_route_fixture_v1_retry2')
OUT = ROOT / 'refactor_v2/reports' / ATTEMPT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def author_primitives():
    manifest = json.loads((ROOT / 'refactor_v2/reports/lapa_author_source_v1/source_and_fixture_audit.json').read_text())
    paths = [SOURCE / 'm3ae/modules/language_encoders/bert_model.py', SOURCE / 'm3ae/modules/m3ae_utils.py']
    assert all(sha(p) == manifest['dependencies'][str(p)] for p in paths)
    wanted = {'clones', 'LayerNorm', 'attention', 'MultiHeadedAttention', 'cross_attention',
              'cross_attention_without_residual', 'poswise_ffd', 'GraphAttentionLayer', 'GAT',
              'LP_fusion_module', 'pre_training_module', 'GAT_module'}
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'copy': copy}
    utils = ast.parse(paths[1].read_text())
    init = next(n for n in utils.body if isinstance(n, ast.FunctionDef) and n.name == 'init_weights')
    exec(compile(ast.Module(body=[init], type_ignores=[]), str(paths[1]), 'exec'), namespace)
    nodes = [n for n in ast.parse(paths[0].read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(paths[0]), 'exec'), namespace)
    return namespace


class SyntheticRoute(nn.Module):
    def __init__(self):
        super().__init__()
        author = author_primitives()
        self.latent = nn.Parameter(torch.randn(32, 32) / 32)
        self.answer_tokens = nn.Parameter(torch.randn(41, 32))
        self.answer_projection = nn.Linear(41, 128)
        self.gm = author['pre_training_module'](32)
        self.lf = nn.ModuleList([author['LP_fusion_module'](32) for _ in range(6)])
        self.pf = author['GAT_module'](32)
        self.organ_pool = nn.Conv1d(577, 4, 1)
        self.output_pool = nn.Conv1d(36, 1, 1)
        self.head = nn.Linear(32, 3)
        self.answer_projection.apply(author['init_weights'])
        self.gm.apply(author['init_weights'])
        self.lf.apply(author['init_weights'])
        self.organ_pool.apply(author['init_weights'])
        self.output_pool.apply(author['init_weights'])

    def forward(self, text, image, graph, adj):
        batch = text.shape[0]
        bank = self.answer_projection(self.answer_tokens.T).T.unsqueeze(0).expand(batch, -1, -1)
        prompt = self.gm(self.latent.unsqueeze(0).expand(batch, -1, -1), bank)
        consistency = prompt.mean(dim=1)
        # Fixed synthetic features, not author cross-modal encoders/poolers.
        for layer in self.lf:
            prompt = layer(prompt, text, image, torch.cat([text, image], dim=1))
        graph_prompt = self.pf(prompt, graph, adj)
        fused = self.output_pool(torch.cat([prompt, self.organ_pool(graph_prompt)], dim=1)).squeeze(1).tanh()
        fused = fused + .1 * text.mean(dim=1) + .1 * image.mean(dim=1)
        return self.head(fused), consistency


def synthetic_inputs():
    generator = torch.Generator().manual_seed(20261025)
    return (torch.randn(2, 7, 32, generator=generator), torch.randn(2, 9, 32, generator=generator),
            torch.randn(2, 577, 32, generator=generator), torch.eye(577).expand(2, -1, -1))


def replay():
    model = SyntheticRoute().eval()
    model.load_state_dict(torch.load(OUT / 'synthetic_state.pt', map_location='cpu'), strict=True)
    with torch.no_grad():
        output = model(*synthetic_inputs())[0]
    torch.save(output, OUT / 'fresh_output.pt')


def main():
    OUT.mkdir(exist_ok=False)
    torch.manual_seed(20261025)
    model = SyntheticRoute()
    inputs = synthetic_inputs()
    initial = {k: v.detach().clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    gradients = {}
    for _ in range(2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, generated = model(*inputs)
        target = torch.zeros_like(logits); target[:, 0] = 1
        answer_target = inputs[0].mean(dim=1)  # Synthetic vector, not invented clinical labels.
        cs = (1 - F.cosine_similarity(generated, answer_target)).sum()
        loss = F.binary_cross_entropy_with_logits(logits, target) * 3 + .1 * cs
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            gradients[name] = max(gradients.get(name, 0), float(parameter.grad.abs().max()))
        optimizer.step()
    for prefix in ('latent', 'answer_', 'gm.', 'lf.', 'pf.', 'organ_pool.', 'output_pool.', 'head.'):
        names = [name for name in initial if name.startswith(prefix)]
        assert names and any(gradients[name] > 0 for name in names), prefix
        assert any(not torch.equal(initial[name], dict(model.named_parameters())[name]) for name in names), prefix
    model.eval()
    with torch.no_grad():
        output = model(*inputs)[0]
        altered_text = model(inputs[0] + torch.randn_like(inputs[0]), *inputs[1:])[0]
        altered_graph = model(inputs[0], inputs[1], inputs[2] + torch.randn_like(inputs[2]), inputs[3])[0]
        assert not torch.equal(output, altered_text)
        assert not torch.equal(output, altered_graph)
    torch.save(model.state_dict(), OUT / 'synthetic_state.pt')
    subprocess.run([sys.executable, str(Path(__file__).resolve()), 'replay'], check=True)
    assert torch.equal(output, torch.load(OUT / 'fresh_output.pt', map_location='cpu'))
    report = {
        'scope': 'Synthetic prompt-only harness: exact pinned GM/LF/GAT primitives; toy32-dimensional features and3class dummy head. No author encoder/pooler, clinical GT, VQA quality run, calibration or training release.',
        'updates': 2, 'finite_gradients': True, 'intended_component_updates': True,
        'strict_fresh_process_replay_exact': True,
        'text_and_graph_interventions_change_output': True,
        'latent_tokens': 32, 'answer_projection_slots': 128, 'LF_blocks': 6, 'graph_slots': 577,
        'GAT_heads': 1, 'attention_heads': 8,
        'source_GAT_nonedge_behavior_preserved': True,
        'clinical_or_native_reproduction_certified': False,
        'gradient_max': gradients,
        'artifacts': {str(p): sha(p) for p in (OUT / 'synthetic_state.pt', OUT / 'fresh_output.pt')},
        'source_sha256': sha(Path(__file__)),
    }
    with (OUT / 'fixture_receipt.json').open('x') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ('gradient_max', 'artifacts')}, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(1)
    if len(sys.argv) > 1:
        replay()
    else:
        main()
