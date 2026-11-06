"""Actual direct/wrapped loss equality and eval bypass, no optimizer updates."""
import json,hashlib,gc,time
from pathlib import Path
import torch
from src.models.bonevqa import BoneVQAModel
from src.data import build_dataset,collate_fn
from src.train import build_split,seed_training_epoch
from src.utils import set_seed,normalize_answer
from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss
from experiments.grouped_native_executor_contract_v3_1 import integration
from experiments.run_vl_bounded_pilot_v3_1 import check_environment
ROOT=Path('/Users/uyen/Downloads/BONEVQA');OUT=ROOT/'refactor_v2/reports/grouped_native_executor_contract_v3_1'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
check_environment();assert not (OUT/'integration_progress.json').exists()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
dump(OUT/'integration_progress.json',{'status':'running'})
start=time.perf_counter();results={}
try:
    contract=read(OUT/'report.json')
    assert contract['source_sha256']==sha(ROOT/'refactor_v2/experiments/grouped_native_executor_contract_v3_1.py')
    assert contract['bindings_sha256']==sha(OUT/'bindings.json')
    fixture=read(ROOT/'refactor_v2/reports/grouped_native_model_gate_v3_2/plan.json')
    for f,binding in read(OUT/'bindings.json').items():
        assert sha(binding['config'])==binding['config_sha256'] and sha(binding['initial_path'])==binding['initial_sha256']
        cfg=read(binding['config']);ds=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
        batch=collate_fn([ds[i] for i in fixture['targets'][f]['indices']])
        raw=read(cfg['data']['loader_kwargs']['sealed_records_by_split']['train'])
        table={(r['question'],normalize_answer(r['answer'])):1. for r in raw if r['answer_type']=='OPEN'}
        set_seed(42);m=BoneVQAModel.load(binding['initial_path'],device='cpu')
        hashes=lambda:{k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in m.trainable_state_dict().items()}
        before=hashes();m.train();seed_training_epoch(42,0)
        with torch.no_grad(),private_training_loss(m,batch,3,'token_mean',table):direct=m(batch,stage=3)
        values={k:float(direct[k]) for k in ('loss','loss_answer','loss_lm')};del direct
        stats={'TRAIN_calls':0,'TRAIN_rows':0,'native_eval_rows':0}
        m.train();seed_training_epoch(42,0)
        with integration(table,stats,binding):
            with torch.no_grad():wrapped=m(batch,stage=3)
            assert values=={k:float(wrapped[k]) for k in values};del wrapped
            m.eval()
            with torch.no_grad():evaluated=m(batch,stage=3)
            assert torch.isfinite(evaluated['loss']);del evaluated
        assert stats=={'TRAIN_calls':1,'TRAIN_rows':4,'native_eval_rows':4}
        assert hashes()==before and sha(binding['initial_path'])==binding['initial_sha256']
        results[f]={'passed':True,'losses':values,'direct_wrapped_exact':True,'native_eval_bypass':True,'weights_unchanged':True,'statistics':stats}
        print('Fold',f,'integration passed',flush=True)
        del m;gc.collect()
    dump(OUT/'model_integration.json',{'passed':True,'verifier_sha256':sha(__file__),'bindings_sha256':sha(OUT/'bindings.json'),'results':results,'elapsed_seconds':time.perf_counter()-start,'optimizer_updates':0,'reserved_role_inference':False,'training_released':False})
    dump(OUT/'integration_progress.json',{'status':'passed'})
except Exception as e:
    dump(OUT/'integration_failure.json',{'error':repr(e),'preserved':True});raise
