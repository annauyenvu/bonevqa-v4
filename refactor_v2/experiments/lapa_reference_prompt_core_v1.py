"""Pinned-author prompt core, awaiting a verified backbone/asset integration.

Unlike the earlier toy harness, each LF call requires its own intermediate
language/image/multimodal features. This module never manufactures them.
"""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.lapa_author_route_fixture_v1 import author_primitives


class ReferencePromptCore(nn.Module):
    def __init__(self, answer_token_count, d_model=768, graph_nodes=577, n_layers=6, n_latent=32):
        super().__init__()
        if answer_token_count < 1 or d_model % 8 or n_layers != 6 or n_latent != 32:
            raise ValueError('Reference requires a nonempty token bank,8-head compatible width,six LF blocks,32prompts')
        author = author_primitives()
        self.d_model, self.graph_nodes = d_model, graph_nodes
        self.latent = nn.Parameter(torch.randn(n_latent, d_model) / d_model)
        self.answer_projection = nn.Linear(answer_token_count, 128)
        self.generation = author['pre_training_module'](d_model)
        self.LF = nn.ModuleList([author['LP_fusion_module'](d_model) for _ in range(n_layers)])
        self.PF = author['GAT_module'](d_model)
        self.graph_reduction = nn.Conv1d(graph_nodes, 4, 1)
        self.prompt_reduction = nn.Conv1d(n_latent, 1, 1)
        self.combined_reduction = nn.Conv1d(n_latent + 4, 1, 1)
        for module in (self.answer_projection, self.generation, self.LF, self.graph_reduction,
                       self.prompt_reduction, self.combined_reduction):
            module.apply(author['init_weights'])

    def generate(self, answer_word_embeddings, batch_size):
        if answer_word_embeddings.shape != (self.answer_projection.in_features, self.d_model):
            raise ValueError('Expected raw answer-word embeddings [TRAINtoken_count,width], not pooled answers')
        bank = self.answer_projection(answer_word_embeddings.T).T.unsqueeze(0).expand(batch_size, -1, -1)
        prompts = self.generation(self.latent.unsqueeze(0).expand(batch_size, -1, -1), bank)
        return prompts, prompts.mean(dim=1)

    def forward(self, answer_word_embeddings, layer_contexts, graph_embeddings=None, adjacency=None, pf_mode='off'):
        if len(layer_contexts) != 6:
            raise ValueError('Provide six actual per-layer (mid_text,mid_image,fused_multimodal) contexts')
        if pf_mode not in ('off', 'author', 'graph_bypass'):
            raise ValueError('Unknown PF mode')
        batch = layer_contexts[0][0].shape[0]
        prompt, consistency_vector = self.generate(answer_word_embeddings, batch)
        for block, contexts in zip(self.LF, layer_contexts):
            if len(contexts) != 3:
                raise ValueError('LF context must contain language,image,multimodal tensors')
            if any(x.ndim != 3 or x.shape[0] != batch or x.shape[-1] != self.d_model for x in contexts):
                raise ValueError('Each context must have shape [batch,tokens,width]')
            prompt = block(prompt, *contexts)
        if pf_mode == 'off':
            summary = self.prompt_reduction(prompt).squeeze(1).tanh()
        else:
            if graph_embeddings is None or graph_embeddings.shape != (batch, self.graph_nodes, self.d_model):
                raise ValueError('Explicit graph embeddings with verified node order required')
            if pf_mode == 'author':
                if adjacency is None or adjacency.shape[-2:] != (self.graph_nodes, self.graph_nodes):
                    raise ValueError('Explicit adjacency required')
                graph_prompt = self.PF(prompt, graph_embeddings, adjacency)
            else:
                # A new, disclosed control: NO graph message passing. Self-only
                # author adjacency is not a graph-off control (see fixture).
                graph_prompt = self.PF.cross_attn_kw(graph_embeddings, prompt, prompt)
            graph_slots = self.graph_reduction(graph_prompt)
            summary = self.combined_reduction(torch.cat([prompt, graph_slots], dim=1)).squeeze(1).tanh()
        return {'latent_summary': summary, 'generated_mean': consistency_vector, 'lf_prompts': prompt}

    @staticmethod
    def author_consistency(generated_mean, target_raw_word_embeddings):
        """Unmasked mean of raw target word embeddings, batch SUM, as released.

        Padding/token construction must be frozen by the future asset adapter.
        This method deliberately does not pretend contextual/masked means match.
        """
        target = target_raw_word_embeddings.mean(dim=1)
        return (1 - F.cosine_similarity(generated_mean, target)).sum()
