"""Public image/question-only greedy generation on the accepted TRAIN gate.

Explicit tokenizer/config ID reconciliation, no targets in generation API.
"""
import json
from pathlib import Path
import time
import torch
from experiments.favp_trainable_route_v3 import TrainableRoute, ROOT
from experiments.lapa_integrated_gate_v1 import batch_data
from experiments.prepare_lapa_pilot_v1 import sha
from src.utils import normalize_answer

OUT=ROOT/'reports/favp_public_generation_v1'


def public_generate(model,image,question):
    # Intentionally no answer, ROI, label, or annotation argument.
    query=model.encode(image)
    tokenizer=model.tokenizer
    instruction=('###Human: <Img><ImageHere></Img> [vqa] Based on the image, '
                 'respond to this question with a short answer: '+question.lower()+'  ###Assistant: ')
    left,right=instruction.split('<ImageHere>')
    encode=lambda s:tokenizer(s,add_special_tokens=False,return_tensors='pt').input_ids
    embed=model.language.get_input_embeddings()
    prefix=torch.cat([embed(torch.tensor([[tokenizer.bos_token_id]])),embed(encode(left)),
                      model.projection(query.float()).half(),embed(encode(right))],1)
    # Checkpoint config IDs 0/1 differ from tokenizer semantic BOS/EOS 1/2.
    # Align to tokenizer explicitly before observing generated outputs.
    outputs=model.language.generate(inputs_embeds=prefix,attention_mask=torch.ones(prefix.shape[:2],dtype=torch.long),
        max_new_tokens=20,min_length=1,do_sample=False,num_beams=1,use_cache=True,
        bos_token_id=tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.,length_penalty=1.)
    tokens=outputs[0].tolist()
    decoded=tokenizer.decode(outputs[0],skip_special_tokens=True)
    answer=decoded.split('###')[0].replace('<s>','').split('Assistant: ')[-1].strip()
    return {'tokens':tokens,'raw_decoded':decoded,'answer':answer,'normalized_answer':normalize_answer(answer),
            'prefix_tokens':prefix.shape[1]}


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(8);torch.manual_seed(42)
    start=time.perf_counter()
    gate=ROOT/'reports/favp_trainable_route_gate_v3'
    replay=json.loads((gate/'replay.json').read_text())
    checkpoint=gate/'checkpoint.pt'
    assert replay['exact_logits'] and replay['exact_loss'] and replay['checkpoint_sha256']==sha(checkpoint)
    model=TrainableRoute()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    mutable={n for n,p in model.named_parameters(remove_duplicate=False) if p.requires_grad}
    assert set(saved['mutable'])==mutable
    for name,h in saved['sources'].items():assert sha(Path(name))==h
    full=model.state_dict();full.update(saved['mutable']);model.load_state_dict(full,strict=True)
    del full,saved
    model.eval()
    batch,_=batch_data()
    rows=[]
    with torch.no_grad():
        for i in [0,1,0]:
            tick=time.perf_counter()
            result=public_generate(model,batch['images'][i],batch['questions'][i])
            rows.append({'image_id':batch['image_ids'][i],'question':batch['questions'][i],**result,'seconds':time.perf_counter()-tick})
    assert rows[0]['tokens']==rows[2]['tokens']
    receipt={'public_inputs':['original image','question'],'no_target_argument':True,'greedy_repeat_exact':True,
        'generation_policy':{'max_new_tokens':20,'beams':1,'sample':False,'EOS':'tokenizer semantic EOS; explicit reconciliation'},
        'checkpoint_config_ids':{'BOS':model.language.config.bos_token_id,'EOS':model.language.config.eos_token_id,'PAD':model.language.config.pad_token_id},
        'effective_tokenizer_ids':{'BOS':model.tokenizer.bos_token_id,'EOS':model.tokenizer.eos_token_id,'PAD':model.tokenizer.pad_token_id},
        'rows':rows,'checkpoint_sha256':sha(checkpoint),'total_seconds':time.perf_counter()-start,
        'scope':'two fixed TRAIN gate examples; post-disposable update inference plumbing, not held-out quality or native stage handoff',
        'quality_released':False,'script_sha256':sha(Path(__file__))}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
