"""Two fixed actual originals through native Lens's public question-only route."""
import json
from pathlib import Path
import resource
import time
import torch
from experiments.lens_native_common_v1 import ROOT, sha, fixed_train_image
from experiments.gate_lens_native_trainability_v1 import build, state_hash

OUT=ROOT/'reports/lens_public_generation_v1'


def public_generate(model,processor,image,question):
    # Deliberately accepts neither labels, region metadata, automatic views nor ROI.
    content=[{'type':'text','text':'Answer briefly using the medical image.'},{'type':'image'},
             {'type':'text','text':question}]
    prompt=processor.apply_chat_template([{'role':'user','content':content}],add_generation_prompt=True).strip()
    inputs=processor(text=[prompt],images=[[image]],return_tensors='pt',padding=True)
    assert int(inputs['pixel_values'].shape[0])==1
    before=time.perf_counter()
    with torch.no_grad():
        tokens=model.generate(**inputs,do_sample=False,max_new_tokens=12,use_cache=True,
            bos_token_id=processor.tokenizer.bos_token_id,eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id)
    generated=tokens[0,inputs['input_ids'].shape[1]:].tolist()
    return {'token_ids':generated,'text':processor.tokenizer.decode(generated,skip_special_tokens=True),
            'prompt_token_count':inputs['input_ids'].shape[1],'image_tile_count':inputs['pixel_values'].shape[1],
            'seconds':time.perf_counter()-before}


def main():
    OUT.mkdir(exist_ok=False)
    accepted=json.loads((ROOT/'reports/lens_native_trainability_gate_v1/replay.json').read_text())
    assert accepted['exact']
    start=time.perf_counter()
    model,processor,*_=build()
    checkpoint=ROOT/'reports/lens_native_trainability_gate_v1/mutable.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert set(state)==set(trainable)
    with torch.no_grad():
        for n,p in trainable.items():p.copy_(state[n])
    assert state_hash(model)==accepted['full_state_digest']
    model.eval()
    results=[]
    # Same fixed TRAIN technical cases. This is not a validation accuracy evaluation.
    question='Is there a fracture in this X-ray?'
    for index in [8,5]:
        image,binding=fixed_train_image(index)
        result=public_generate(model,processor,image,question)
        repeat=public_generate(model,processor,image,question)
        assert result['token_ids']==repeat['token_ids']
        results.append({'image':binding,'question':question,'generation':result,'exact_token_repeat':True})
        (OUT/'progress.json').write_text(json.dumps({'completed':len(results),'results':results},indent=2))
    receipt={'public_input_contract':'original PIL image + question only; no target/region/ROI/view arguments',
        'cases':results,'max_new_tokens':12,'decoding':'greedy','checkpoint_sha256':sha(checkpoint),
        'checkpoint_scope':'disposable two-update technical gate, not pilot initialization',
        'script_sha256':sha(Path(__file__)),'total_seconds':time.perf_counter()-start,
        'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'tokenizer_semantic_ids':{k:getattr(processor.tokenizer,k) for k in ['bos_token_id','eos_token_id','pad_token_id']},
        'quality_released':False,'scope':'technical public original-only deterministic generation, no quality/clinical claim'}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
