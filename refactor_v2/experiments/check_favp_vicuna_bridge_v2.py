"""Real SAM/EVA/QFormer feature → new projection → native Vicuna causal loss."""
import json
import resource
import time
from pathlib import Path
import torch
from experiments.favp_vicuna_loader_v2 import load_vicuna
from experiments.favp_visual_bridge_v2 import ROOT
from experiments.lapa_integrated_gate_v1 import batch_data
from experiments.prepare_lapa_pilot_v1 import sha

OUT=ROOT/'reports/favp_vicuna_bridge_v2'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(8);torch.manual_seed(42);start=time.perf_counter()
    source=ROOT/'reports/favp_visual_bridge_v2/features.pt'
    saved=torch.load(source,map_location='cpu',weights_only=True)
    batch,answers=batch_data();assert batch['image_ids'][0]==saved['image_id']
    model,tokenizer,audit=load_vicuna()
    projection=torch.nn.Linear(768,4096).half().eval()
    query=saved['queries'].half()
    instruction='###Human: <Img><ImageHere></Img> [vqa] Based on the image, respond to this question with a short answer: '+batch['questions'][0]+'  ###Assistant: '
    left,right=instruction.split('<ImageHere>')
    encode=lambda x:tokenizer(x,add_special_tokens=False,return_tensors='pt').input_ids
    embedding=model.get_input_embeddings()
    with torch.no_grad():
        prefix=torch.cat([embedding(torch.tensor([[tokenizer.bos_token_id]])),embedding(encode(left)),projection(query),embedding(encode(right))],1)
        answer_ids=encode(answers[0]+'###')
        sequence=torch.cat([prefix,embedding(answer_ids)],1)
        targets=torch.full(sequence.shape[:2],-100,dtype=torch.long);targets[:,prefix.shape[1]:]=answer_ids
        tick=time.perf_counter();result=model(inputs_embeds=sequence,attention_mask=torch.ones(sequence.shape[:2],dtype=torch.long),
            labels=targets,use_cache=False,return_dict=True)
        seconds=time.perf_counter()-tick
    assert torch.isfinite(result.loss) and torch.isfinite(result.logits).all()
    assert (targets[:,:prefix.shape[1]]==-100).all() and (targets!=-100).sum()==answer_ids.numel()
    receipt={'image_id':saved['image_id'],'feature_sha256':sha(source),'audit':audit,
        'prefix_tokens':prefix.shape[1],'supervised_answer_tokens':answer_ids.numel(),'sequence_tokens':sequence.shape[1],
        'loss':float(result.loss),'forward_seconds':seconds,'total_seconds':time.perf_counter()-start,
        'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope':'native author LM forward with actual image query features and random new projector; teacher-forced TRAIN answer loss only, no optimizer/generation/QA quality',
        'stage_handoff':'missing verified FAVP-to-Vicuna projection; current projection random is a disclosed technical bridge',
        'quality_released':False,'sources':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('favp_vicuna_loader_v2.py')]}}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(e)},f)
        raise

