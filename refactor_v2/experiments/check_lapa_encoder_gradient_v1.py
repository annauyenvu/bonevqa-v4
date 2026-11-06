"""Disposable real-TRAIN answer gradient gate; no quality training."""
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.check_lapa_real_features_v1 import ROOT, REPORTS, sha
from experiments.lapa_interleaved_reference_v1 import InterleavedReference
from src.models.encoders import VisualEncoder, TextEncoder
from src.data import build_dataset, collate_fn
from src.train import build_split
from src.utils import normalize_answer

OUT = REPORTS / 'lapa_encoder_gradient_v1'


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    started = time.perf_counter()
    binding = json.loads((REPORTS/'lapa_port_backbone_binding_v1/cache_binding.json').read_text())
    for model in binding['models']:
        for file in model['files']:
            assert sha(Path(file['resolved_path'])) == file['sha256']
    checkpoint = REPORTS/'step4_localization_v3_1/B_PREtask.pt'
    release = json.loads((REPORTS/'vnext_local_tokens_v1/training_release.json').read_text())
    assert sha(checkpoint) == release['dependencies'][str(checkpoint)]
    state = torch.load(checkpoint, map_location='cpu')['state']
    visual = VisualEncoder('biomedclip', 768, lora_rank=4, image_size=224)
    text = TextEncoder(visual, 768, max_len=48)
    supplied = {k[7:]:v for k,v in state.items() if k.startswith('visual.')}
    result = visual.load_state_dict(supplied, strict=False)
    assert not result.unexpected_keys
    assert all(torch.equal(v, visual.state_dict()[k]) for k,v in supplied.items())
    del state
    cfg = json.loads((REPORTS/'step4_localization_v3_1/B.json').read_text())
    ds = build_split(build_dataset, cfg['data']['dataset'], 'train', None, True, cfg['data']['loader_kwargs'])
    indices = json.loads((REPORTS/'vnext_local_tokens_v1/gate_indices.json').read_text())[:2]
    batch = collate_fn([ds[i] for i in indices])
    vocab = json.loads((REPORTS/'step4_localization_v3_1/vocab.json').read_text())
    ids = [t for answer in vocab for t in text.tokenizer.encode(answer, add_special_tokens=True)]
    # Explicit disposable classifier endpoint; not the historical generative head.
    labels = torch.tensor([vocab.index(normalize_answer(ds[i]['answer'])) for i in indices])
    target = F.one_hot(labels, len(vocab)).float()
    text.eval(); visual.eval()
    with torch.no_grad():
        encoded = text(batch['questions'], torch.device('cpu'))
        bank = text.model.get_input_embeddings()(torch.tensor(ids))
    adapter = InterleavedReference(len(ids)).eval()
    head = torch.nn.Linear(768, len(vocab))
    tokens = visual(visual.preprocess(batch['images'], torch.device('cpu')))['tokens']
    output = adapter(encoded['tokens'], tokens, encoded['mask'], bank)
    loss = F.binary_cross_entropy_with_logits(head(output['features']), target)*len(vocab)
    loss.backward()
    groups = {
        'visual_lora': [p for n,p in visual.named_parameters() if 'lora_' in n],
        'crossmodal': list(adapter.text_layers.parameters())+list(adapter.image_layers.parameters()),
        'GM': list(adapter.prompt.generation.parameters()),
        'LF': list(adapter.prompt.LF.parameters()),
        'head': list(head.parameters()),
    }
    stats = {}
    for name, params in groups.items():
        grads = [p.grad for p in params if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), name
        maximum = max(g.abs().max().item() for g in grads)
        assert maximum > 0, name
        stats[name] = {'gradient_tensors':len(grads), 'max_absolute_gradient':maximum}
    receipt = {'scope':'One backward pass on two TRAIN records, class-scaled BCE, disposable random linear classifier; no optimizer or performance experiment.',
               'image_ids':batch['image_ids'], 'loss':loss.item(), 'gradients':stats,
               'elapsed_seconds':time.perf_counter()-started,
               'cache_bytes_verified':True, 'cache_runtime_access_traced':False,
               'quality_release':False, 'updates_or_reload_verified':False,
               'script_sha256':sha(Path(__file__))}
    with (OUT/'receipt.json').open('x') as f: json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))


if __name__ == '__main__':
    try: main()
    except Exception as error:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f: json.dump({'error':repr(error)},f)
        raise
