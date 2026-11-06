"""Two-TRAIN-record inference bridge; no optimizer, targets, fitting or download."""
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from experiments.lapa_interleaved_reference_v1 import InterleavedReference
from src.data import build_dataset, collate_fn
from src.train import build_split
from src.models.encoders import TextEncoder, VisualEncoder

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'refactor_v2/reports'
ATTEMPT = os.environ.get('BONEVQA_REAL_FEATURE_ATTEMPT', 'lapa_real_features_gate_v1')
assert ATTEMPT in ('lapa_real_features_gate_v1', 'lapa_real_features_gate_v1_retry2')
OUT = REPORTS / ATTEMPT


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(8); torch.manual_seed(42)
    started = time.perf_counter()
    checkpoint = REPORTS / 'step4_localization_v3_1/B_PREtask.pt'
    release = json.loads((REPORTS / 'vnext_local_tokens_v1/training_release.json').read_text())
    expected = release['dependencies'][str(checkpoint)]
    assert sha(checkpoint) == expected
    cfg = json.loads((REPORTS / 'step4_localization_v3_1/B.json').read_text())
    loaded = torch.load(checkpoint, map_location='cpu')
    state = loaded['state']
    visual = VisualEncoder('biomedclip', 768, lora_rank=4, image_size=224).eval()
    text = TextEncoder(visual, 768, max_len=48).eval()
    visual_state = {k[len('visual.'):]:v for k,v in state.items() if k.startswith('visual.')}
    text_state = {k[len('text.'):]:v for k,v in state.items() if k.startswith('text.')}
    # BoneVQA.save stores ONLY trainable canonical parameters, not frozen base
    # weights or duplicate aliases. Check every supplied tensor, never silently
    # substitute an incompatible key/shape. Frozen base comes from offline cache.
    visual_load = visual.load_state_dict(visual_state, strict=False)
    text_load = text.load_state_dict(text_state, strict=False)
    assert not visual_load.unexpected_keys and not text_load.unexpected_keys
    assert visual_state and any('lora_' in k for k in visual_state)
    assert all(torch.equal(v, visual.state_dict()[k]) for k,v in visual_state.items())
    assert all(torch.equal(v, text.state_dict()[k]) for k,v in text_state.items())
    partial_load = {'visual_supplied':len(visual_state),'text_supplied':len(text_state),
                    'visual_missing_frozen_or_alias_keys':len(visual_load.missing_keys),
                    'text_missing_frozen_or_alias_keys':len(text_load.missing_keys),
                    'all_supplied_tensors_exact':True,'full_frozen_backbone_equality_to_checkpoint_proven':False}
    del loaded, state; gc.collect()
    dataset = build_split(build_dataset, cfg['data']['dataset'], 'train', None, True, cfg['data']['loader_kwargs'])
    indices = json.loads((REPORTS/'vnext_local_tokens_v1/gate_indices.json').read_text())[:2]
    batch = collate_fn([dataset[i] for i in indices])
    vocab = json.loads((REPORTS/'step4_localization_v3_1/vocab.json').read_text())
    assert isinstance(vocab, list) and len(vocab) == 41
    # Explicit port preprocessing: concatenate TRAIN-vocabulary word-piece IDs
    # including per-answer special tokens. Author preprocessed bank unavailable.
    bank_ids = [token for answer in vocab for token in text.tokenizer.encode(answer, add_special_tokens=True)]
    with torch.no_grad():
        image_tokens = visual(visual.preprocess(batch['images'], torch.device('cpu')))['tokens']
        encoded = text(batch['questions'], torch.device('cpu'))
        bank = text.model.get_input_embeddings()(torch.tensor(bank_ids))
    encode_seconds = time.perf_counter() - started
    adapter = InterleavedReference(len(bank_ids)).eval()
    start = time.perf_counter()
    with torch.no_grad():
        result = adapter(encoded['tokens'], image_tokens, encoded['mask'], bank)
        control = adapter(encoded['tokens'], image_tokens, encoded['mask'], bank, include_prompt=False)
    assert result['features'].shape == (2,768) and torch.isfinite(result['features']).all()
    assert torch.equal(result['carrier'], control['features'])
    timings = {'backbone_load_and_encode_seconds':encode_seconds,'reference_and_control_forward_seconds':time.perf_counter()-start,'total_seconds':time.perf_counter()-started}
    torch.save({'features':result['features'],'carrier':result['carrier'],'generated_mean':result['generated_mean']},OUT/'outputs.pt')
    report = {
        'scope':'Two actual TRAIN records, original images/questions, offline BiomedCLIP/PubMedBERT port. No clinical targets read into forward, no optimizer/fitting/calibration/quality training.',
        'train_image_ids':batch['image_ids'],'record_indices':indices,'TRAIN_vocab_count':len(vocab),'raw_bank_token_count':len(bank_ids),
        'bank_preprocessing':'Concatenated per-answer word pieces including special tokens; disclosed port, not author preprocessed-bank equality.',
        'upstream_checkpoint_sha256':expected,'partial_adapter_load':partial_load,
        'image_token_shape':list(image_tokens.shape),'text_token_shape':list(encoded['tokens'].shape),
        'trace':result['trace'],'finite_output':True,'zero_prompt_control_carrier_exact_eval':True,
        'output_shape':list(result['features'].shape),'timings':timings,
        'new_crossmodal_and_prompt_weights':'Random author-style initialization, NOT M3AE pretrained tensors.',
        'backbone_substitution':'BiomedCLIP/PubMedBERT224, NOT original LaPA backbone/tokenizer/checkpoint.',
        'PF_integrated':False,'visual_encoder_gradient_gate_passed':False,'full_public_runner_integrated':False,
        'native_reproduction_or_quality_released':False,
        'dependencies':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('lapa_interleaved_reference_v1.py'),REPORTS/'step4_localization_v3_1/vocab.json',REPORTS/'step4_localization_v3_1/train.json']},
        'output_sha256':sha(OUT/'outputs.pt'),
    }
    with (OUT/'inference_bridge_receipt.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(error),'preserve_all_outputs':True,'quality_training_started':False},f,indent=2)
        raise
