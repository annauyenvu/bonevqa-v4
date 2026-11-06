"""LaPA PF mechanism with the sealed reconstructed SLAKE graph.

Node names use mean raw word embeddings, excluding padding/special tokens.
This explicit adaptation does not claim equality to the missing author assets.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from experiments.lapa_author_route_fixture_v1 import author_primitives

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / 'refactor_v2/reports/pf_slake_graph_v1'


def load_graphs():
    seal = json.loads((GRAPH / 'graph_seal.json').read_text())
    for path, expected in seal['artifacts'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Graph seal mismatch: {path}')
    nodes = json.loads((GRAPH / 'nodes.json').read_text())
    assert len(nodes) == 409 and len({n['key'] for n in nodes}) == 409
    graphs = {}
    for mode, filename in [('real', 'real_graph.json'), ('rewired', 'degree_placebo.json')]:
        edges = json.loads((GRAPH / filename).read_text())['message_edges']
        adjacency = torch.zeros(len(nodes), len(nodes))
        for i, j in edges:
            assert 0 <= i < len(nodes) and 0 <= j < len(nodes)
            adjacency[i, j] = 1
        assert torch.equal(adjacency, adjacency.T) and adjacency.diag().all()
        graphs[mode] = adjacency
    assert torch.equal(graphs['real'].sum(0), graphs['rewired'].sum(0))
    return nodes, graphs


class SlakeReferencePF(nn.Module):
    def __init__(self, tokenizer, width=768):
        super().__init__()
        nodes, graphs = load_graphs()
        self.nodes = nodes
        encoded = tokenizer([n['label'] for n in nodes], add_special_tokens=False,
                            padding=True, truncation=False, return_tensors='pt')
        assert encoded['attention_mask'].sum(1).min() > 0
        self.register_buffer('node_ids', encoded['input_ids'])
        self.register_buffer('node_mask', encoded['attention_mask'].bool())
        for mode, adjacency in graphs.items():
            self.register_buffer('adj_' + mode, adjacency)
        author = author_primitives()
        self.pf = author['GAT_module'](width)
        self.graph_pool = nn.Conv1d(409, 4, 1)
        self.combined_pool = nn.Conv1d(36, 1, 1)
        self.graph_pool.apply(author['init_weights'])
        self.combined_pool.apply(author['init_weights'])

    def forward(self, prompts, word_embedding, mode='real'):
        if mode not in ('real', 'rewired', 'bypass'):
            raise ValueError('Expected real, rewired or bypass')
        if prompts.ndim != 3 or prompts.shape[1] != 32:
            raise ValueError('Expected 32 LF prompts')
        words = word_embedding(self.node_ids)
        mask = self.node_mask.unsqueeze(-1)
        nodes = (words * mask).sum(1) / mask.sum(1)
        nodes = nodes.unsqueeze(0).expand(prompts.shape[0], -1, -1)
        if mode == 'bypass':
            graph_prompt = self.pf.cross_attn_kw(nodes, prompts, prompts)
        else:
            graph_prompt = self.pf(prompts, nodes, getattr(self, 'adj_' + mode))
        slots = self.graph_pool(graph_prompt)
        return self.combined_pool(torch.cat((prompts, slots), dim=1)).squeeze(1).tanh()
