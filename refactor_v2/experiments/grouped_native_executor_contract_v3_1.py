"""Capped native training integration. This module alone grants no execution release."""
import copy,time
from pathlib import Path
from contextlib import contextmanager

def check_save(path,extra,binding):
    assert extra['stage']==3 and extra['epoch'] in (1,2)
    expected=sum(binding['epoch_updates'][:extra['epoch']])
    assert extra['global_step']==expected<=binding['total_updates']
    assert Path(path).resolve().parent==Path(binding['checkpoint_dir']).resolve()

def check_counts(statistics,binding):
    assert 0<=statistics['TRAIN_calls']<=binding['max_TRAIN_calls']
    assert 0<=statistics['TRAIN_rows']<=binding['max_TRAIN_rows']
    assert 0<=statistics['native_eval_rows']<=binding['max_eval_rows']

@contextmanager
def integration(table,statistics,binding):
    from src.models.bonevqa import BoneVQAModel
    from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss
    old_forward,old_save=BoneVQAModel.forward,BoneVQAModel.save
    start=time.monotonic()
    def forward(self,batch,stage=2):
        assert time.monotonic()-start<=binding['seconds_cap']
        assert stage==3
        train=bool(self.training);n=len(batch['questions'])
        prospective=dict(statistics)
        if train:
            prospective['TRAIN_calls']+=1;prospective['TRAIN_rows']+=n
        else:prospective['native_eval_rows']+=n
        check_counts(prospective,binding)
        with private_training_loss(self,batch,stage,'token_mean',table) as trace:
            result=old_forward(self,batch,stage=stage)
        if not train:assert not trace['applied'] and trace['calls']==0
        statistics.update(prospective)
        return result
    def save(self,path,extra=None):
        assert binding.get('checkpoint_writes_released') is True
        extra=copy.deepcopy(extra or {});check_save(path,extra,binding)
        extra['grouped_native_binding']=dict(binding)
        return old_save(self,path,extra)
    BoneVQAModel.forward,BoneVQAModel.save=forward,save
    try:yield
    finally:BoneVQAModel.forward,BoneVQAModel.save=old_forward,old_save

def prepare():
    import json,hashlib
    root=Path('/Users/uyen/Downloads/BONEVQA');prep=root/'refactor_v2/reports/grouped_native_preparation_v3_2'
    out=root/'refactor_v2/reports/grouped_native_executor_contract_v3_1';assert not out.exists()
    read=lambda p:json.loads(p.read_text())
    def sha(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    assert read(prep/'independent_audit.json')['passed']
    gate=root/'refactor_v2/reports/grouped_native_export_gate_v3_1/report.json';assert read(gate)['passed']
    out.mkdir();bindings={};rejected=0
    for fold in ('1','2','3','4'):
        cp=prep/f'fold_{fold}.json';sp=prep/f'fold_{fold}_schedule.json';c=read(cp);s=read(sp)
        train=read(Path(c['data']['loader_kwargs']['sealed_records_by_split']['train']))
        val=read(Path(c['data']['loader_kwargs']['sealed_records_by_split']['model_val']))
        b={'fold':fold,'config':str(cp),'config_sha256':sha(cp),'schedule_sha256':sha(sp),
           'initial_path':c['init_from'],'initial_sha256':sha(Path(c['init_from'])),
           'checkpoint_dir':str(root/c['checkpoint_dir']/c['run_name']),
           'epoch_updates':[e['updates'] for e in s],'total_updates':sum(e['updates'] for e in s),
           'max_TRAIN_calls':sum(len(e['ordered_indices']) for e in s),'max_TRAIN_rows':2*len(train),
           'max_eval_rows':2*len(val),'seconds_cap':10800,'checkpoint_writes_released':False}
        for epoch in (1,2):check_save(Path(b['checkpoint_dir'])/'last.pt',{'stage':3,'epoch':epoch,'global_step':sum(b['epoch_updates'][:epoch])},b)
        for bad in ({'stage':3,'epoch':3,'global_step':b['total_updates']},{'stage':3,'epoch':2,'global_step':b['total_updates']+1}):
            try:check_save(Path(b['checkpoint_dir'])/'last.pt',bad,b)
            except AssertionError:rejected+=1
            else:raise RuntimeError('Overrun accepted')
        try:check_counts({'TRAIN_calls':b['max_TRAIN_calls']+1,'TRAIN_rows':0,'native_eval_rows':0},b)
        except AssertionError:rejected+=1
        else:raise RuntimeError('Extra forward accepted')
        bindings[fold]=b
    (out/'bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
    (out/'report.json').write_text(json.dumps({'source_sha256':sha(Path(__file__)),'bindings_sha256':sha(out/'bindings.json'),'passed_cap_fixtures':True,'negative_cases_rejected':rejected,'checkpoint_gate_report_sha256':sha(gate),'actual_model_integration_pending':True,'training_released':False},indent=2)+'\n')
    print('Four fold bindings prepared;12 overrun cases rejected; actual integration pending')

if __name__=='__main__':prepare()
