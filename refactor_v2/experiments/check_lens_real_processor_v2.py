"""Actual local SmolVLM processor and native visual features on SAM views."""
import json
from pathlib import Path
import resource
import sys
import time
import torch
from PIL import Image
from transformers import AutoProcessor,AutoConfig,Idefics3ForConditionalGeneration
from experiments.lens_native_common_v1 import ROOT,SOURCE,sha,verify_source,verify_package,fixed_train_image

OUT=ROOT/'reports/lens_real_processor_v2'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(8);torch.manual_seed(42)
    start=time.perf_counter()
    sam=json.loads((ROOT/'reports/lens_sam_native_recovery_v2/receipt.json').read_text())
    assert sam['recovered_view_artifacts'] and sam['retained_masks_after_source_filters']>0
    for p,h in sam['view_artifacts'].items():assert sha(Path(p))==h
    image,binding=fixed_train_image(8);assert binding==sam['image']
    directory=ROOT/'reports/lens_sam_native_v1/views'
    views=[image,Image.open(directory/'sam_unicolor.png').convert('RGB'),Image.open(directory/'sam_masked.png').convert('RGB')]
    folder=verify_package('SmolVLM-256M-Instruct')
    processor=AutoProcessor.from_pretrained(folder,local_files_only=True)
    verify_source('src/localization_lens/modeling.py')
    sys.path.insert(0,str(SOURCE/'src'))
    from localization_lens.modeling import pool_semantic_views
    content=[{'type':'text','text':'Answer briefly using the medical image.'}]+[{'type':'image'} for _ in views]+[{'type':'text','text':'What is visible in the image?'}]
    text=processor.apply_chat_template([{'role':'user','content':content}],add_generation_prompt=True)
    batch=processor(text=[text],images=[views],return_tensors='pt',padding=True)
    pixels=batch['pixel_values'];attention=batch.get('pixel_attention_mask')
    assert pixels.ndim==5 and pixels.shape[0]==1
    counts=[];separate=[]
    for view in views:
        individual=processor.image_processor(images=[[view]],return_tensors='pt')
        value=individual['pixel_values'];valid=value.ne(0).any(dim=(-1,-2,-3))[0]
        counts.append(int(valid.sum()));separate.append(value[:,valid])
    joined=torch.cat(separate,dim=1)
    real=pixels.ne(0).any(dim=(-1,-2,-3))[0]
    assert torch.equal(joined,pixels[:,real]),'Combined processor changed semantic-view tile ordering'
    count=int(real.sum());assert count==sum(counts) and count<=128
    image_tokens=int((batch['input_ids']==processor.image_token_id).sum()) if hasattr(processor,'image_token_id') else int((batch['input_ids']==49190).sum())
    assert image_tokens==count*64
    config=AutoConfig.from_pretrained(folder,local_files_only=True)
    old_pad=getattr(config,'pad_token_id',None);config.pad_token_id=config.text_config.pad_token_id
    model,loading=Idefics3ForConditionalGeneration.from_pretrained(folder,config=config,local_files_only=True,
        use_safetensors=True,dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
    assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading.get('mismatched_keys',[])
    model.eval().requires_grad_(False)
    tick=time.perf_counter()
    with torch.no_grad():
        result=model.model.get_image_features(pixel_values=pixels,pixel_attention_mask=attention)
        features=result.pooler_output
        pooled=pool_semantic_views(features,pixels,num_views=3)
        expected=[];cursor=0
        for n in counts:
            expected.append(features[cursor:cursor+n].mean(dim=(-2,-3)))
            cursor+=n
        expected=torch.stack(expected)[None]
        assert torch.isfinite(features).all() and torch.isfinite(pooled).all()
        # Exact source pooling is only meaningful where actual tile counts agree.
        equal_counts=len(set(counts))==1
        if equal_counts:assert torch.allclose(pooled,expected,atol=1e-6,rtol=1e-5)
        repeat=model.model.get_image_features(pixel_values=pixels,pixel_attention_mask=attention).pooler_output
        assert torch.equal(features,repeat)
    torch.save({'features':features,'pooled':pooled,'per_view_tile_counts':counts},OUT/'visual_features.pt')
    receipt={'image':binding,'native_automatic_masks':sam['retained_masks_after_source_filters'],
        'view_types':['original','native_SAM_unicolor','native_SAM_masked'],'actual_per_view_tile_counts':counts,
        'pixel_shape':list(pixels.shape),'image_token_count':image_tokens,'source_image_seq_len':64,
        'combined_tile_order_exact':True,'equal_tiles_observed':equal_counts,'feature_shape':list(features.shape),
        'pooled_shape':list(pooled.shape),'finite_exact_repeat':True,'source_pool_max_difference_vs_provenance':float((pooled-expected).abs().max()),
        'original_top_pad_id':old_pad,'effective_pad_id':config.pad_token_id,
        'loading_info':{k:sorted(v) if isinstance(v,set) else v for k,v in loading.items()},
        'two_visual_forwards_seconds':time.perf_counter()-tick,'total_seconds':time.perf_counter()-start,
        'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'quality_released':False,
        'scope':'one real native automatic-SAM three-view case through actual pinned processor and pretrained SmolVLM visual connector; not LM loss/backward/full Lens or general equal-tile proof',
        'script_sha256':sha(Path(__file__))}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise

