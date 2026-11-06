"""Author-code cross-modal/LF ordering with an explicitly external backbone.

No backbone download, native-reproduction claim or automatic quality execution.
"""
import ast
from pathlib import Path

import torch
from torch import nn
from transformers import BertConfig
from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer

from experiments.lapa_author_route_fixture_v1 import SOURCE, author_primitives
from experiments.lapa_reference_prompt_core_v1 import ReferencePromptCore


def cross_layer_primitives():
    namespace = author_primitives()  # verifies pinned author source hashes
    namespace.update({'ACT2FN': ACT2FN, 'apply_chunking_to_forward': apply_chunking_to_forward,
                      'find_pruneable_heads_and_indices': find_pruneable_heads_and_indices,
                      'prune_linear_layer': prune_linear_layer})
    names = {'BertSelfAttention', 'BertSelfOutput', 'BertAttention', 'BertIntermediate', 'BertOutput', 'BertCrossLayer'}
    path = SOURCE / 'm3ae/modules/language_encoders/bert_model.py'
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name in names]
    assert {n.name for n in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    pool_path = SOURCE / 'm3ae/modules/prediction_heads.py'
    pool_node = next(n for n in ast.parse(pool_path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Pooler')
    exec(compile(ast.Module(body=[pool_node], type_ignores=[]), str(pool_path), 'exec'), namespace)
    return namespace


class InterleavedReference(nn.Module):
    def __init__(self, answer_token_count, d_model=768, mm_heads=12, graph_nodes=577):
        super().__init__()
        author = cross_layer_primitives()
        cfg = BertConfig(hidden_size=d_model, num_hidden_layers=6, num_attention_heads=mm_heads,
                         intermediate_size=4*d_model, hidden_dropout_prob=.1, attention_probs_dropout_prob=.1)
        self.text_projection = nn.Linear(d_model, d_model)
        self.image_projection = nn.Linear(d_model, d_model)
        self.modality_types = nn.Embedding(2, d_model)
        self.text_layers = nn.ModuleList([author['BertCrossLayer'](cfg) for _ in range(6)])
        self.image_layers = nn.ModuleList([author['BertCrossLayer'](cfg) for _ in range(6)])
        self.prompt = ReferencePromptCore(answer_token_count, d_model=d_model, graph_nodes=graph_nodes)
        self.text_pool = author['Pooler'](d_model)
        self.image_pool = author['Pooler'](d_model)
        for module in (self.text_projection, self.image_projection, self.modality_types,
                       self.text_layers, self.image_layers, self.text_pool, self.image_pool):
            module.apply(author['init_weights'])

    def forward(self, text_tokens, image_tokens, text_mask, answer_word_embeddings, include_prompt=True):
        if text_tokens.ndim != 3 or image_tokens.ndim != 3 or text_mask.shape != text_tokens.shape[:2]:
            raise ValueError('Invalid external-backbone token/mask shapes')
        batch = text_tokens.shape[0]
        x = self.text_projection(text_tokens) + self.modality_types(torch.zeros_like(text_mask, dtype=torch.long))
        y = self.image_projection(image_tokens) + self.modality_types(torch.ones(image_tokens.shape[:2], dtype=torch.long, device=x.device))
        text_pad = (1-text_mask[:, None, None, :].to(x.dtype))*torch.finfo(x.dtype).min
        image_pad = torch.zeros(batch, 1, 1, y.shape[1], dtype=y.dtype, device=y.device)
        latent, generated_mean = self.prompt.generate(answer_word_embeddings, batch)
        trace = []
        for index, (text_layer, image_layer, lf) in enumerate(zip(self.text_layers, self.image_layers, self.prompt.LF)):
            # Both streams read the same PREVIOUS x,y, exactly as author infer.
            mid_text, new_text = text_layer(x, y, text_pad, image_pad, output_attentions=True)
            mid_image, new_image = image_layer(y, x, image_pad, text_pad, output_attentions=True)
            x, y = new_text[0], new_image[0]
            latent = lf(latent, mid_text, mid_image, torch.cat([x, y], dim=1))
            trace.extend([f'MM{index+1}', f'LF{index+1}'])
        # This public component currently implements GM+LF only. Real PF assets
        # and their verified order are required before adding native PF.
        prompt_summary = self.prompt.prompt_reduction(latent).squeeze(1).tanh()
        carrier = .1*self.text_pool(x) + .1*self.image_pool(y)
        output = carrier + prompt_summary if include_prompt else carrier
        return {'features': output, 'carrier': carrier, 'generated_mean': generated_mean,
                'lf_prompts': latent, 'trace': trace}
