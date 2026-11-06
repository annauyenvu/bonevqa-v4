"""Versioned, native-asset Lens route for the prospectively fixed bounded pilot."""
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import torch
from PIL import Image
from transformers import AutoConfig,AutoProcessor,Idefics3ForConditionalGeneration
from peft import LoraConfig,TaskType,get_peft_model
from experiments.lens_native_common_v1 import ROOT,SOURCE,sha,verify_source,verify_package
from experiments.gate_lens_native_trainability_v1 import tensor_hash,state_hash

INPUTS=ROOT/'reports/lens_bounded_inputs_v1'
PREP=ROOT/'reports/lens_bounded_preparation_v1'
QUALITY=ROOT/'reports/lens_bounded_quality_v1'


def source_functions():
    for name in ['training.py','modeling.py','inference.py']:
        verify_source('src/localization_lens/'+name)
    if str(SOURCE/'src') not in sys.path:sys.path.insert(0,str(SOURCE/'src'))
    from localization_lens.training import LensCollator
    from localization_lens.modeling import pool_semantic_views,DecoupledContrastiveLoss,InfoNCELoss
    from localization_lens.inference import normalize_answer
    return LensCollator,pool_semantic_views,DecoupledContrastiveLoss,InfoNCELoss,normalize_answer


def load_native():
    torch.set_num_threads(8);torch.manual_seed(42)
    folder=verify_package('SmolVLM-256M-Instruct')
    processor=AutoProcessor.from_pretrained(folder,local_files_only=True)
    processor.image_processor.size={'longest_edge':1024}
    config=AutoConfig.from_pretrained(folder,local_files_only=True)
    config.pad_token_id=config.text_config.pad_token_id
    model,loading=Idefics3ForConditionalGeneration.from_pretrained(folder,config=config,local_files_only=True,
        use_safetensors=True,dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
    assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading.get('mismatched_keys',[])
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
        target_modules=['q_proj','k_proj','v_proj','o_proj','down_proj','up_proj','gate_proj'],task_type=TaskType.CAUSAL_LM))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    return model,processor


def trainable(model):return {n:p for n,p in model.named_parameters() if p.requires_grad}


def restore(model,path):
    state=torch.load(path,map_location='cpu',weights_only=True);params=trainable(model)
    assert set(state)==set(params)
    with torch.no_grad():
        for n,p in params.items():
            assert p.shape==state[n].shape and p.dtype==state[n].dtype
            p.copy_(state[n])


def save_mutable(model,path):torch.save({n:p.detach().cpu().clone() for n,p in trainable(model).items()},path)


def source_rows(role):
    path=ROOT/f'reports/step4_localization_v3_1/{role}.json'
    seal=json.loads((path.parent/'data_seal.json').read_text())
    assert sha(path)==seal['artifacts'][f'refactor_v2/reports/step4_localization_v3_1/{role}.json']
    return json.loads(path.read_text())


def inputs():
    receipt=json.loads((INPUTS/'receipt.json').read_text());assert receipt['complete']
    for p,h in receipt['dependencies'].items():assert sha(Path(p))==h,p
    cases={r['image']['image_id']:r for r in receipt['cases'] if r['eligible']}
    return receipt,[cases[i] for i in receipt['selected_training_ids']]


def example(case,position,phase):
    row=source_rows('train')[case['candidate']['record_index']]
    assert row['image_id']==case['image']['image_id'] and row['region']==case['region']
    original=Path(case['image']['path']);assert sha(original)==case['image']['file_sha256']
    kinds=random.Random(42+position).sample(sorted(['unicolor','multicolor','masked']),2)
    paths=[Path(case['view_directory'])/('sam_'+k+'.png') for k in kinds]
    for p in paths:assert sha(p)==case['artifacts'][str(p)]
    images=[Image.open(original).convert('RGB')]+[Image.open(p).convert('RGB') for p in paths]
    return {'images':images,'question':'Which anatomical region is emphasized by these localization views?' if phase=='align' else row['question'],
            'answer':case['region'] if phase=='align' else row['answer']}


def make_batch(processor,cases,indices,phase,check_pixels=False):
    Collator,*_=source_functions();collator=Collator(processor)
    assert len(indices)==2 and len(set(indices))==2
    examples=[example(cases[i],i,phase) for i in indices]
    batch=collator(examples)
    counts=[]
    for i,item in enumerate(examples):
        prefix=processor.apply_chat_template(collator._messages(item,False),add_generation_prompt=True).strip()
        prefix_ids=processor(text=prefix,images=item['images'],return_tensors='pt')['input_ids'][0]
        assert torch.equal(batch['input_ids'][i,:len(prefix_ids)],prefix_ids),'Source collation/padding mismatch'
        assert (batch['labels'][i,:len(prefix_ids)]==-100).all()
        assert (batch['labels'][i][batch['input_ids'][i]==49190]==-100).all()
        assert (batch['labels'][i][batch['attention_mask'][i]==0]==-100).all()
        assert int((batch['labels'][i]!=-100).sum())>0
        per_view=[];separate=[]
        for view in item['images']:
            processed=processor.image_processor(images=[[view]],return_tensors='pt')['pixel_values']
            real=processed.ne(0).any(dim=(-1,-2,-3))[0]
            per_view.append(int(real.sum()))
            if check_pixels:separate.append(processed[:,real])
        assert len(set(per_view))==1 and 0<sum(per_view)<=30
        real=batch['pixel_values'][i].ne(0).any(dim=(-1,-2,-3))
        assert int(real.sum())==sum(per_view)
        assert int((batch['input_ids'][i]==49190).sum())==sum(per_view)*64
        if check_pixels:assert torch.equal(torch.cat(separate,dim=1)[0],batch['pixel_values'][i,real])
        counts.append(per_view)
    return batch,{'indices':indices,'per_view_tile_counts':counts,'pixels':list(batch['pixel_values'].shape),
                  'tokens':list(batch['input_ids'].shape),'answer_tokens':(batch['labels']!=-100).sum(dim=1).tolist()}


def objective(model,batch,phase,enabled):
    _,pool,DCL,NCE,_=source_functions()
    output=model(**batch,output_hidden_states=enabled)
    ce=output.loss
    if not enabled:return ce,{'CE':float(ce.detach()),'contrastive':0.,'enabled':False},None
    image=model.get_base_model().model.get_image_features(pixel_values=batch['pixel_values'],
        pixel_attention_mask=batch.get('pixel_attention_mask')).pooler_output
    pooled=pool(image,batch['pixel_values'],num_views=3)
    mask=batch['labels'].ne(-100).unsqueeze(-1)
    text=(output.hidden_states[-1]*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)
    contrastive=sum(DCL(.1)(pooled[:,v],text) for v in range(3))/3 if phase=='align' else NCE(.1)(pooled.mean(dim=1),text)
    loss=ce+.1*contrastive
    return loss,{'CE':float(ce.detach()),'contrastive':float(contrastive.detach()),'enabled':True},(pooled,text,contrastive)


def optimizer_and_scheduler(model,total):
    optimizer=torch.optim.AdamW(trainable(model).values(),lr=2e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:max(0.,1-step/total))
    return optimizer,scheduler


def gradient_audit(model):
    groups={}
    for n,p in model.named_parameters():
        if not p.requires_grad:assert p.grad is None,n;continue
        assert p.grad is not None and torch.isfinite(p.grad).all(),n
        key='vision' if 'vision_model' in n else 'text'
        groups[key]=groups.get(key,0.)+float(p.grad.abs().sum())
    assert set(groups)=={'vision','text'} and min(groups.values())>0
    norm=torch.nn.utils.clip_grad_norm_(trainable(model).values(),1.)
    assert torch.isfinite(norm)
    return {'gradient_l1':groups,'preclip_norm':float(norm)}


def public_generation(model,processor,image,question):
    # Exact source inference text template; no target or annotation arguments.
    content=[{'type':'text','text':'Answer briefly using the medical image.'},{'type':'image'},{'type':'text','text':question}]
    prompt=processor.apply_chat_template([{'role':'user','content':content}],add_generation_prompt=True)
    batch=processor(text=[prompt],images=[[image]],return_tensors='pt',padding=True)
    with torch.no_grad():
        output=model.generate(**batch,do_sample=False,max_new_tokens=32,use_cache=True,
            bos_token_id=processor.tokenizer.bos_token_id,eos_token_id=processor.tokenizer.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id)
    tokens=output[0,batch['input_ids'].shape[1]:].tolist()
    return {'tokens':tokens,'text':processor.tokenizer.decode(tokens,skip_special_tokens=True).strip(),
            'truncated':len(tokens)==32 and tokens[-1]!=processor.tokenizer.eos_token_id,
            'input_ids_sha256':tensor_hash(batch['input_ids']),'tiles':int(batch['pixel_values'].shape[1])}


def metrics(rows):
    _,_,_,_,normalize=source_functions()
    correct=[];leading=[];tp=tn=fp=fn=unsupported=0
    for row in rows:
        target=normalize(row['target']);prediction=normalize(row['text'])
        assert target in ('yes','no')
        correct.append(prediction==target)
        match=re.match(r'^\s*(yes|no)\b',row['text'],flags=re.I)
        parsed=match.group(1).lower() if match else None
        leading.append(parsed==target);unsupported+=parsed is None
        tp+=target=='yes' and parsed=='yes';tn+=target=='no' and parsed=='no'
        fn+=target=='yes' and parsed!='yes';fp+=target=='no' and parsed!='no'
    sens=tp/(tp+fn) if tp+fn else None;spec=tn/(tn+fp) if tn+fp else None
    return {'records':len(rows),'strict_accuracy':sum(correct)/len(rows),'leading_binary_accuracy':sum(leading)/len(rows),
        'secondary_unsupported':unsupported,'truncated':sum(r['truncated'] for r in rows),'empty':sum(not r['text'] for r in rows),
        'tp':tp,'tn':tn,'fp_or_unsupported_negative':fp,'fn_or_unsupported_positive':fn,'sensitivity':sens,'specificity':spec,
        'balanced_accuracy':(sens+spec)/2 if sens is not None and spec is not None else None}


def evaluate(model,processor,selection,directory,stage,updates):
    import time
    model.eval();targets=source_rows('model_val');rows=[];start=time.perf_counter()
    for selected in selection:
        row=targets[selected['record_index']]
        assert row['image_id']==selected['image_id'] and row['question']==selected['question']
        path=ROOT.parent/'.evaluation_runtime/data/fracatlas/FracAtlas'/row['image']
        generated=public_generation(model,processor,Image.open(path).convert('RGB'),row['question'])
        rows.append({'image_id':row['image_id'],'image_sha256':sha(path),'question':row['question'],'target':row['answer'],**generated})
        (directory/(stage+'_progress.json')).write_text(json.dumps({'completed':len(rows),'seconds':time.perf_counter()-start}))
    result={'stage':stage,'updates':updates,'rows':rows,'metrics':metrics(rows),'seconds':time.perf_counter()-start}
    (directory/(stage+'.json')).write_text(json.dumps(result,indent=2))
    return result
