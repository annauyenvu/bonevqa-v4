"""Materialize P consistency/bank factorial configurations without model fitting."""
import copy,hashlib,json
from pathlib import Path
import torch
from src.checkpointing import checkpoint_manifest,validate_manifest
from src.utils import normalize_answer

ROOT=Path(__file__).resolve().parents[2];R=ROOT/'refactor_v2/reports';P=R/'step4_localization_v3_1';H=R/'step56_screening_v3_1';O=R/'pnext_consistency_bank_v1_prepare2'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2)
def hashes(state):return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in state.items()}
def main():
    assert not O.exists();O.mkdir()
    try:
        closure=read(R/'step6_confirmation_v3_1/steps56_closure.json')
        for p,h in closure['artifacts'].items():assert sha(Path(p))==h,p
        for p,h in read(H/'execution_release.json')['dependencies'].items():assert sha(Path(p))==h,p
        parent=torch.load(H/'P_real_PREtask.pt',map_location='cpu',weights_only=False)
        random=torch.load(H/'P_random_PREtask.pt',map_location='cpu',weights_only=False)
        baseline=torch.load(P/'B_PREtask.pt',map_location='cpu',weights_only=False)
        validate_manifest(parent,parent['config'],parent['answer_vocab']);ref=hashes(parent['state'])
        assert hashes(random['state'])==ref
        shared=hashes(baseline['state']);assert all(ref[k]==v for k,v in shared.items())
        assert parent['extra']['zero_QA_updates'] and random['extra']['zero_QA_updates']
        answers={normalize_answer(r['answer']) for r in read(P/'train.json')}
        assert {normalize_answer(a) for a in parent['answer_vocab']}==answers
        assert parent['config']['eta']==.01 and parent['config']['prior_labels']==[]
        plan={'cells':{},'new_arms':[],'quality_released':False,'wait_for':str(R/'lnext_spatial_runs_v1/analysis_closure.json'),'epochs':4,'updates':292,'seed':42,'shared_B_tensors_exact':len(shared),'P_tensors_exact':len(ref),'TRAIN_vocab_only_verified':True,'parent_initialization_sha256':sha(H/'P_real_PREtask.pt'),'source_files':{str(p):sha(p) for p in (Path(__file__),R/'PNEXT_CONSISTENCY_BANK_PROTOCOL_20261025.md')}}
        for bank in ('train','random','empty'):
            for on in (True,False):
                arm='P_'+bank+'_cs'+str(int(on));old='P_real' if bank=='train' else 'P_random'
                if on and bank!='empty':
                    folder=H/('run_'+old);replay=read(folder/'full_replay.json');assert replay['passed']
                    source=folder/'epoch4_evaluation.json';assert sha(source)==read(H/'screening_selection.json')['dependencies'][str(source)]
                    plan['cells'][arm]={'reused':True,'old_arm':old,'evaluation':str(source),'evaluation_sha256':sha(source),'replay':str(folder/'full_replay.json'),'replay_sha256':sha(folder/'full_replay.json')}
                    continue
                payload=copy.deepcopy(parent);payload['config'].update(latent_answer_bank_mode=bank,eta=.01 if on else 0.)
                payload['extra']={'zero_QA_updates':True,'unchanged_tensor_parent':str(H/'P_real_PREtask.pt'),'parent_sha256':sha(H/'P_real_PREtask.pt'),'interventions':{'bank':bank,'consistency_on':on}}
                payload['manifest']=checkpoint_manifest(payload['config'],payload['answer_vocab'],payload['state'])
                validate_manifest(payload,payload['config'],payload['answer_vocab']);assert hashes(payload['state'])==ref
                init=O/(arm+'_PREtask.pt');torch.save(payload,init)
                cfg=read(H/'P_real.json');cfg['model']=copy.deepcopy(payload['config']);cfg['init_from']=str(init);cfg['run_name']=arm
                cfg['pnext']={'arm':arm,'bank':bank,'consistency_on':on,'eta':.01 if on else 0.,'epochs':4,'updates':292,'seed':42}
                write(O/(arm+'.json'),cfg)
                plan['cells'][arm]={'reused':False,'config':str(O/(arm+'.json')),'initialization_sha256':sha(init)};plan['new_arms'].append(arm)
        write(O/'plan.json',plan)
        files=list(O.glob('*.json'))+list(O.glob('*.pt'))+[H/'P_real_PREtask.pt',H/'P_random_PREtask.pt',H/'execution_release.json',H/'screening_selection.json']+[P/n for n in ('train.json','model_val.json','schedules.json','data_seal.json','vocab.json','component_lookup.json')]
        write(O/'preparation_seal.json',{'artifacts':{str(p):sha(p) for p in files},'model_gates_pending':True,'quality_released':False})
        print(json.dumps({'prepared_cells':len(plan['cells']),'new_arms':plan['new_arms'],'reused_arms':2,'shared_B_tensors_exact':len(shared),'model_gates_pending':True}))
    except Exception as e:
        write(O/'preparation_failure.json',{'error':repr(e),'preserve_all_outputs':True});raise
if __name__=='__main__':main()
