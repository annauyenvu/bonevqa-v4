"""Actual two-image native Lens CE/DCL/InfoNCE updates and fresh replay."""
import csv
import gc
import hashlib
import json
import random
import resource
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Idefics3ForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model
from experiments.lens_native_common_v1 import ROOT, SOURCE, sha, verify_source, verify_package, fixed_train_image

OUT = ROOT/'reports/lens_native_trainability_gate_v1'
PROTOCOL = ROOT/'reports/LENS_NATIVE_TRAINABILITY_GATE_PROTOCOL_V1.md'


def write(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2))


def tensor_hash(tensor):
    t = tensor.detach().cpu().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def state_hash(model, frozen_only=False):
    h = hashlib.sha256()
    trainable = {n for n,p in model.named_parameters() if p.requires_grad}
    for n,t in sorted(model.state_dict().items()):
        if frozen_only and n in trainable:
            continue
        h.update(n.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(bytes.fromhex(tensor_hash(t)))
    return h.hexdigest()


def make_examples(phase, processor):
    receipt = json.loads((ROOT/'reports/lens_fixed_native_views_v1/receipt.json').read_text())
    assert receipt['technical_batch_indices'] == [8,5]
    rows = json.loads((ROOT/'reports/step4_localization_v3_1/train.json').read_text())
    metadata = ROOT.parent/'.evaluation_runtime/data/fracatlas/FracAtlas/dataset.csv'
    with metadata.open(encoding='utf-8-sig') as f:
        csv_rows = {r['image_id']:r for r in csv.DictReader(f)}
    examples, bindings = [], []
    for j,index in enumerate([8,5]):
        case = next(c for c in receipt['cases'] if c['index']==index)
        image,binding = fixed_train_image(index); assert binding==case['image']
        row = rows[index]; meta = csv_rows[row['image_id']+'.jpg']
        regions = [r for r in ['hand','leg','hip','shoulder'] if int(meta[r])==1]
        region = ' and '.join(regions)
        assert region and region==row['region'], 'Unverified region supervision'
        chosen = random.Random(42+j).sample(sorted(['unicolor','multicolor','masked']),2)
        paths = [Path(case['view_directory'])/('sam_'+v+'.png') for v in chosen]
        # Accepted artifact bindings, not newly generated views.
        artifacts = case.get('artifacts') or json.loads((ROOT/'reports/lens_sam_native_recovery_v2/receipt.json').read_text())['view_artifacts']
        for p in paths: assert sha(p)==artifacts[str(p)]
        views = [image]+[Image.open(p).convert('RGB') for p in paths]
        question = 'Which anatomical region is emphasized by these localization views?' if phase=='align' else row['question']
        answer = region if phase=='align' else row['answer']
        examples.append({'images':views,'question':question,'answer':answer})
        bindings.append({'image':binding,'view_types':['original']+chosen,'view_hashes':{str(p):sha(p) for p in paths},
                         'answer':answer,'question':question,'metadata_row':meta,'metadata_sha256':sha(metadata)})
    assert bindings[0]['image']['file_sha256']!=bindings[1]['image']['file_sha256']
    return examples,bindings


def build():
    torch.set_num_threads(8); torch.manual_seed(42)
    for rel in ['src/localization_lens/training.py','src/localization_lens/modeling.py']: verify_source(rel)
    sys.path.insert(0,str(SOURCE/'src'))
    from localization_lens.training import LensCollator
    from localization_lens.modeling import pool_semantic_views, DecoupledContrastiveLoss, InfoNCELoss
    folder = verify_package('SmolVLM-256M-Instruct')
    processor = AutoProcessor.from_pretrained(folder,local_files_only=True)
    processor.image_processor.size = {'longest_edge':1024}
    assert processor.image_processor.max_image_size=={'longest_edge':512}
    config = AutoConfig.from_pretrained(folder,local_files_only=True)
    config.pad_token_id = config.text_config.pad_token_id
    model,loading = Idefics3ForConditionalGeneration.from_pretrained(folder,config=config,local_files_only=True,
        use_safetensors=True,dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
    assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading.get('mismatched_keys',[])
    model = get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
        target_modules=['q_proj','k_proj','v_proj','o_proj','down_proj','up_proj','gate_proj'],task_type=TaskType.CAUSAL_LM))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    batches,bindings = {},{}
    collator = LensCollator(processor)
    for phase in ['align','vqa']:
        examples,bindings[phase] = make_examples(phase,processor)
        batch = collator(examples)
        counts=[]
        for i,item in enumerate(examples):
            separate=[]; per_view=[]
            for view in item['images']:
                value=processor.image_processor(images=[[view]],return_tensors='pt')['pixel_values']
                valid=value.ne(0).any(dim=(-1,-2,-3))[0]
                separate.append(value[:,valid]); per_view.append(int(valid.sum()))
            real=batch['pixel_values'][i].ne(0).any(dim=(-1,-2,-3))
            assert torch.equal(torch.cat(separate,dim=1)[0],batch['pixel_values'][i,real])
            assert len(set(per_view))==1 and sum(per_view)<=30
            assert int((batch['input_ids'][i]==config.image_token_id).sum())==sum(per_view)*64
            prefix=processor.apply_chat_template(collator._messages(item,False),add_generation_prompt=True).strip()
            prefix_ids=processor(text=prefix,images=item['images'],return_tensors='pt')['input_ids'][0]
            # A source collator left-padding mismatch is a hard failure, not a silent mask correction.
            assert torch.equal(batch['input_ids'][i,:len(prefix_ids)],prefix_ids)
            assert (batch['labels'][i,:len(prefix_ids)]==-100).all()
            assert (batch['labels'][i][batch['input_ids'][i]==config.image_token_id]==-100).all()
            assert (batch['labels'][i]!=-100).sum()>0
            counts.append(per_view)
        batches[phase]=batch
        bindings[phase]={'cases':bindings[phase],'per_view_tile_counts':counts,'pixel_shape':list(batch['pixel_values'].shape),
                         'input_shape':list(batch['input_ids'].shape),'answer_tokens':(batch['labels']!=-100).sum(dim=1).tolist()}
    return model,processor,batches,bindings,pool_semantic_views,DecoupledContrastiveLoss,InfoNCELoss


def eval_signature(model,batch):
    model.eval()
    with torch.no_grad():
        output=model(**batch)
        result={'logits_sha256':tensor_hash(output.logits),'logit_shape':list(output.logits.shape),'loss':float(output.loss)}
    assert torch.isfinite(output.logits).all() and torch.isfinite(output.loss)
    return result


def main():
    replay='--replay' in sys.argv
    if not replay: OUT.mkdir(exist_ok=False)
    start=time.perf_counter()
    model,processor,batches,bindings,pool,DCL,NCE=build()
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    frozen=state_hash(model,True)
    dependencies={str(p):sha(p) for p in [Path(__file__),PROTOCOL,ROOT/'experiments/lens_native_common_v1.py',
        ROOT/'reports/lens_fixed_native_views_v1/receipt.json',ROOT/'reports/lens_runtime_v2/receipt.json',
        verify_source('src/localization_lens/training.py'),verify_source('src/localization_lens/modeling.py')]}
    if replay:
        before=json.loads((OUT/'update.json').read_text())
        assert dependencies==before['dependencies'] and bindings==before['bindings'] and frozen==before['frozen_digest']
        checkpoint=torch.load(OUT/'mutable.pt',map_location='cpu',weights_only=True)
        assert set(checkpoint)==set(params)
        with torch.no_grad():
            for n,p in params.items(): assert p.shape==checkpoint[n].shape; p.copy_(checkpoint[n])
        assert state_hash(model)==before['final_full_state_digest']
        signature=eval_signature(model,batches['vqa'])
        assert signature==before['eval_signature'], (signature,before['eval_signature'])
        write('replay.json',{'exact':True,'signature':signature,'full_state_digest':state_hash(model),
              'seconds':time.perf_counter()-start,'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        return
    initial={n:tensor_hash(p) for n,p in params.items()}
    write('started.json',{'dependencies':dependencies,'bindings':bindings,'trainable_parameters':sum(p.numel() for p in params.values()),
                         'trainable_names':list(params),'initial_full_state_digest':state_hash(model),'quality_released':False})
    optimizer=torch.optim.AdamW(params.values(),lr=2e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    steps=[]
    for step,phase in enumerate(['align','vqa'],1):
        tick=time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
        batch=batches[phase]
        output=model(**batch,output_hidden_states=True)
        features=model.get_base_model().model.get_image_features(pixel_values=batch['pixel_values'],
            pixel_attention_mask=batch.get('pixel_attention_mask')).pooler_output
        pooled=pool(features,batch['pixel_values'],num_views=3)
        mask=batch['labels'].ne(-100).unsqueeze(-1)
        text=(output.hidden_states[-1]*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)
        contrastive=sum(DCL(.1)(pooled[:,v],text) for v in range(3))/3 if phase=='align' else NCE(.1)(pooled.mean(dim=1),text)
        probes=torch.autograd.grad(contrastive,(pooled,text),retain_graph=True)
        assert all(torch.isfinite(g).all() and float(g.abs().sum())>0 for g in probes)
        loss=output.loss+.1*contrastive; assert torch.isfinite(loss)
        loss.backward()
        groups={}
        for n,p in params.items():
            assert p.grad is not None and torch.isfinite(p.grad).all(),n
            group='vision' if 'vision_model' in n else 'text'
            groups[group]=groups.get(group,0.)+float(p.grad.abs().sum())
        assert set(groups)=={'vision','text'} and min(groups.values())>0
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        torch.nn.utils.clip_grad_norm_(params.values(),1.); optimizer.step()
        record={'step':step,'phase':phase,'CE':float(output.loss.detach()),'contrastive':float(contrastive.detach()),
                'total':float(loss.detach()),'gradient_l1_by_group':groups,'contrastive_feature_gradient_l1':[float(g.abs().sum()) for g in probes],
                'seconds':time.perf_counter()-tick}
        steps.append(record);write('step_'+str(step)+'.json',record)
        del output,features,pooled,text,loss,contrastive,probes;gc.collect()
    assert state_hash(model,True)==frozen
    changed=[n for n,p in params.items() if tensor_hash(p)!=initial[n]]
    assert any('vision_model' in n for n in changed) and any('text_model' in n for n in changed)
    torch.save({n:p.detach().cpu() for n,p in params.items()},OUT/'mutable.pt')
    signature=eval_signature(model,batches['vqa'])
    write('update.json',{'steps':steps,'dependencies':dependencies,'bindings':bindings,'frozen_digest':frozen,
        'frozen_invariance':True,'changed_names':changed,'trainable_count':len(params),'eval_signature':signature,
        'final_full_state_digest':state_hash(model),'mutable_sha256':sha(OUT/'mutable.pt'),
        'seconds':time.perf_counter()-start,'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'quality_released':False})


if __name__=='__main__':
    try: main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/('replay_failure.json' if '--replay' in sys.argv else 'failure.json')).open('x') as f:
                json.dump({'error':repr(exc)},f)
        raise
