"""Restricted M3AE text-state load and target-free native RoBERTa inference."""
import json
import time
from pathlib import Path
import torch
from transformers import RobertaConfig,RobertaModel,AutoTokenizer
from experiments.prepare_lapa_pilot_v1 import sha

ROOT=Path(__file__).resolve().parents[2]/'refactor_v2'
OUT=ROOT/'reports/native_roberta_from_m3ae_v1'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(8);start=time.perf_counter()
    path=ROOT/'references/native_assets_lapa_v1/m3ae.ckpt'
    assert sha(path)=='469cf69b18cd75fbc3a010f202fb42c0cc074b34db75a0bfab442db982885442'
    state=torch.load(path,map_location='cpu',weights_only=True)['state_dict']
    prefix='language_encoder.'
    text={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    del state
    folder=ROOT/'references/native_assets_lapa_v1/roberta-base'
    cfg=RobertaConfig.from_pretrained(folder,local_files_only=True)
    with torch.device('meta'):model=RobertaModel(cfg)
    status=model.load_state_dict(text,strict=True,assign=True)
    assert not status.missing_keys and not status.unexpected_keys
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in text.items())
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    questions=['Is there a fracture in this X-ray?','Which body region is shown in this X-ray?']
    inputs=tokenizer(questions,padding=True,return_tensors='pt')
    assert inputs['input_ids'].max()<cfg.vocab_size
    model.eval()
    with torch.no_grad():
        output=model(**inputs).last_hidden_state
        repeat=model(**inputs).last_hidden_state
    assert torch.isfinite(output).all() and torch.equal(output,repeat)
    receipt={'checkpoint_sha256':sha(path),'state_tensors':len(text),'strict_load':True,
             'all_supplied_tensors_exact':True,'shape':list(output.shape),'finite':True,
             'deterministic_repeat':True,'public_inputs':'questions only','seconds':time.perf_counter()-start,
             'scope':'Native RoBERTa encoder weights extracted from restricted-load M3AE; no image/MM/prompt model or QA head executed',
             'quality_released':False,'script_sha256':sha(Path(__file__))}
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(error)},f)
        raise
