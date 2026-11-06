"""Three bounded T controls on the current ROCO PREtask/B pilot protocol.

Prepare -> fresh-process verify -> serial run. Never executes old PMC configs,
changes frozen sources, consumes reserved roles, retries or starts long training.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

import torch
from src.context_controls import ComponentBatchSampler,load_component_lookup,component_disjoint_donors
from src.data import build_dataset,collate_fn
from src.data.prompts import textual_prompt
from src.models.bonevqa import BoneVQAModel
from src.protocol import manifest,records_of,validate_prompt_protocol_config
from src.train import build_split,seed_training_epoch
from src.utils import set_seed
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree
from experiments.vl_bounded_execution_contract_v3_1 import INIT,INIT_SHA
from experiments.vl_bounded_proposal_contract_v3_1 import sha
from experiments.run_vl_bounded_pilot_v3_1 import check_environment

ROOT=Path.cwd();BASE=ROOT/'refactor_v2/reports/fracatlas_scale_pilot_v3'
FOLDER=ROOT/'refactor_v2/reports/T_bounded_current_v3_1'
STYLES=('clinical','irrelevant_matched','clinical_paraphrase')


def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def hashes(state):return {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in state.items()}


def check_idle():
    commands=subprocess.check_output(['ps','-axo','command'],text=True)
    assert not any(' -m src.train ' in r or ('Python' in r and 'run_vl_bounded_pilot_v3_1.py --run' in r) for r in commands.splitlines()), 'Existing training; do not compete'


def data_and_schedule():
    base=json.loads((BASE/'fracatlas_B_roco_scale_pilot_v3.json').read_text())
    kwargs=base['data']['loader_kwargs']
    assert kwargs['sealed_records_by_split']['calibration'] is None and kwargs['sealed_records_by_split']['test'] is None
    assert not kwargs['with_masks'] and not kwargs['with_boxes'] and kwargs['boxes_from'] is None
    datasets={r:build_split(build_dataset,base['data']['dataset'],r,None,True,kwargs) for r in ('train','model_val')}
    expected=json.loads((BASE/'execution_release.json').read_text())['actual_manifests']
    assert {r:manifest(d) for r,d in datasets.items()}==expected
    assert len(datasets['train'])==1173 and len(datasets['model_val'])==142
    bc=base['train']['component_batching'];lookup,_=load_component_lookup(bc['manifest'],bc['seal'],ROOT)
    sampler=ComponentBatchSampler([lookup[r['image_id']] for r in records_of(datasets['train'])],4,42)
    expected_schedule=json.loads((ROOT/'refactor_v2/reports/vl_bounded_execution_contract_v3_1/component_epoch_schedule.json').read_text())
    for epoch in range(4):
        sampler.set_epoch(epoch);batches=list(sampler)
        assert batches==expected_schedule[epoch]['ordered_indices'] and len(batches)==294
        for batch in batches:
            groups=[lookup[records_of(datasets['train'])[i]['image_id']] for i in batch]
            donors=component_disjoint_donors(groups)
            assert all(groups[i]!=groups[j] for i,j in enumerate(donors))
    return base,datasets,expected


def validate_config(cfg,style,parent):
    assert cfg['seed']==42 and cfg['model']=={**parent['config'],'prompt_variant':'T','use_textual_prompt':True,'textual_prompt_style':style,'use_region_in_prompt':False}
    assert not any(cfg['model'][k] for k in ('use_visual_prompt','use_lens','use_latent_prompt'))
    assert cfg['train']['stage']==3 and cfg['train']['epochs']==4
    assert cfg['train']['batch_size']==4 and cfg['train']['grad_accum']==4 and cfg['train']['lr']==1e-4
    assert cfg['train']['weight_decay']==.01 and cfg['train']['max_grad_norm']==1.
    assert cfg['train']['epoch_seeded_rng'] and cfg['train'].get('warmup_ratio',.05)==.05
    assert not cfg['train']['amp'] and cfg['train']['eval_max_samples'] is None
    validate_prompt_protocol_config(cfg)


def prepare():
    check_environment();check_idle();assert not FOLDER.exists()
    subprocess.run([sys.executable,'refactor_v2/scripts/seal_technical_reference_v3.py','--verify'],check=True)
    base,datasets,manifests=data_and_schedule()
    assert sha(INIT)==INIT_SHA
    parent=torch.load(INIT,map_location='cpu',weights_only=False)
    assert len(parent['state'])==263 and len(parent['answer_vocab'])==68
    questions=sorted({r['question'] for r in records_of(datasets['train'])})
    source_gate=json.loads((ROOT/'refactor_v2/reports/vl_matched_control_audit_v3_1/report.json').read_text())
    batch=collate_fn([datasets['train'][i] for i in source_gate['selected_TRAIN_QA_indices']])
    FOLDER.mkdir();state={'status':'preparing_T_gates','pid':os.getpid(),'started_utc':now()};dump(FOLDER/'preflight_progress.json',state)
    started=time.perf_counter();configs={};initials={};gates={};token_report={}
    try:
        for style in STYLES:
            cfg=copy.deepcopy(base);cfg['model']=copy.deepcopy(parent['config'])
            cfg['model'].update(prompt_variant='T',use_textual_prompt=True,textual_prompt_style=style,use_region_in_prompt=False)
            name='T_'+style+'_seed42_bounded_current_v3_1'
            assert not (ROOT/base['checkpoint_dir']/name).exists()
            cfg.update(run_name=name,recipe_status='candidate_three_T_bounded_ONLY_not_long_release',figure_dir=str(FOLDER/name/'figures'))
            init_path=FOLDER/(name+'_PREtask.pt');cfg['init_from']=str(init_path)
            validate_config(cfg,style,parent)
            set_seed(42);model=BoneVQAModel(cfg['model'],parent['answer_vocab'],device='cpu')
            model.load_state_dict(parent['state'],strict=False)
            assert hashes(model.trainable_state_dict())==hashes(parent['state'])
            model.save(str(init_path),{'scope':'fresh_T_PREtask_zeroQAupdates','parent_ROCO_PREtask_sha256':INIT_SHA})
            initials[style]={'path':str(init_path),'sha256':sha(init_path)}
            if not token_report:
                biomedical=getattr(model.text.tokenizer,'tokenizer',model.text.tokenizer)
                language=model.llm_tokenizer
                for candidate in ('raw',*STYLES):
                    lengths=[]
                    for q in questions:
                        text=q if candidate=='raw' else textual_prompt(q,region=None,style=candidate)
                        assert q in text and text.count(q)==1
                        bio=len(biomedical(text,add_special_tokens=True,truncation=False)['input_ids'])
                        llm=len(language(model._format_llm_prompt(text),add_special_tokens=False,truncation=False)['input_ids'])
                        assert bio<=48 and llm<=56
                        lengths.append({'question':q,'biomedical':bio,'language':llm})
                    token_report[candidate]=lengths
                assert [(r['biomedical'],r['language']) for r in token_report['clinical']]==[(r['biomedical'],r['language']) for r in token_report['irrelevant_matched']]
            for q in questions:
                assert model._question_text(q,'INJECTED_REGION_MUST_BE_IGNORED')==textual_prompt(q,region=None,style=style)
            params=dict(model.named_parameters())
            names=['qformer.query','prefix_proj.0.weight',next(k for k in params if k.startswith('visual.') and 'lora_B' in k)]
            before={k:params[k].detach().clone() for k in names}
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.98),weight_decay=.01)
            gradients=[]
            for step in range(2):
                seed_training_epoch(42,step);model.train();optimizer.zero_grad(set_to_none=True)
                out=model(batch,stage=3)
                values=torch.autograd.grad(out['loss_answer'],[params[k] for k in names],retain_graph=True,allow_unused=True)
                grad={k:0. if v is None else float(v.abs().max()) for k,v in zip(names,values)}
                assert torch.isfinite(out['loss']) and all(v is None or torch.isfinite(v).all() for v in values)
                out['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();gradients.append(grad)
            updates={k:float((params[k].detach()-before[k]).abs().max()) for k in names}
            assert all(v>0 for v in gradients[-1].values()) and all(v>0 for v in updates.values())
            assert sha(init_path)==initials[style]['sha256'] and finite_tree(model.trainable_state_dict())
            gates[style]={'passed':True,'answer_only_gradients':gradients,'selected_actual_updates':updates,'smoke_weights_discarded':True,'T_has_no_extra_parameters':True}
            cfgpath=FOLDER/(name+'.json');dump(cfgpath,cfg);configs[style]={'path':str(cfgpath),'sha256':sha(cfgpath),'run_name':name}
            del model,optimizer,params,before,out,values;gc.collect()
        dump(FOLDER/'TRAIN_token_audit.json',{'unique_questions':len(questions),'rows':1173,'styles':token_report,'clinical_irrelevant_matched_exact_both_tokenizers':True,'all_questions_preserved':True})
        report={'passed':True,'scope':'three seed42 capped T style candidates; not long training or efficacy release',
                'source_script_sha256':sha(__file__),'parent_ROCO_PREtask_sha256':INIT_SHA,'configs':configs,'initials':initials,
                'shared_all263_initial_tensor_hashes':hashes(parent['state']),'answer_vocab':parent['answer_vocab'],
                'actual_manifests':manifests,'TRAIN_gate_indices':source_gate['selected_TRAIN_QA_indices'],'control_gates':gates,
                'token_audit_sha256':sha(FOLDER/'TRAIN_token_audit.json'),'budget_each':'stage3_4epochs296updates1173TRAIN142modelval_CPU8float32',
                'primary':'LAST296','raw_B_reference':'existing seed42 ROCO B LAST296; no best-epoch or best-seed selection',
                'elapsed_seconds':time.perf_counter()-started,'calibration_or_test_used':False,'long_training_authorized':False}
        dump(FOLDER/'preflight_report.json',report);state.update(status='passed_pending_fresh_process_verification');dump(FOLDER/'preflight_progress.json',state)
        print('T token/shared-initialization/answer-gradient/update gates passed; no quality training yet')
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error));dump(FOLDER/'preflight_progress.json',state);raise


def verify():
    check_environment();check_idle();output=FOLDER/'independent_preflight_verification.json';assert not output.exists()
    report=json.loads((FOLDER/'preflight_report.json').read_text());assert report['passed'] and sha(__file__)==report['source_script_sha256']
    assert sha(INIT)==report['parent_ROCO_PREtask_sha256']==INIT_SHA
    parent=torch.load(INIT,map_location='cpu',weights_only=False)
    _,_,actual=data_and_schedule();assert actual==report['actual_manifests']
    for style in STYLES:
        cfg=json.loads(Path(report['configs'][style]['path']).read_text());validate_config(cfg,style,parent)
        assert sha(report['configs'][style]['path'])==report['configs'][style]['sha256']
        init=report['initials'][style];assert sha(init['path'])==init['sha256']
        payload=torch.load(init['path'],map_location='cpu',weights_only=False)
        assert payload['config']==cfg['model'] and payload['answer_vocab']==parent['answer_vocab']
        assert hashes(payload['state'])==report['shared_all263_initial_tensor_hashes']==hashes(parent['state'])
        assert payload['extra']['scope']=='fresh_T_PREtask_zeroQAupdates' and finite_tree(payload['state'])
        assert report['control_gates'][style]['passed'] and report['control_gates'][style]['smoke_weights_discarded']
    assert sha(FOLDER/'TRAIN_token_audit.json')==report['token_audit_sha256']
    dependencies={str(p):sha(p) for p in list((ROOT/'refactor_v2/src').rglob('*.py'))+list(FOLDER.glob('*.json'))+[Path(__file__).resolve()]}
    dump(output,{'passed':True,'preflight_report_sha256':sha(FOLDER/'preflight_report.json'),'dependencies_sha256':dependencies,
                 'all3fresh_initials263tensors_byteexact':True,'zero_quality_updates':True,'long_training_authorized':False})
    print('Fresh process verified T candidates and all263 shared initialization tensors; no quality updates')


def run():
    check_environment();check_idle();assert not (FOLDER/'progress.json').exists()
    verified=json.loads((FOLDER/'independent_preflight_verification.json').read_text());assert verified['passed']
    for path,digest in verified['dependencies_sha256'].items():assert sha(path)==digest
    report=json.loads((FOLDER/'preflight_report.json').read_text());assert sha(FOLDER/'preflight_report.json')==verified['preflight_report_sha256']
    descriptor=os.open(FOLDER/'ACTIVE.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(descriptor)
    state={'status':'starting_serial_bounded_T','pid':os.getpid(),'started_utc':now(),'long_training_authorized':False}
    dump(FOLDER/'progress.json',state)
    try:
        dump(FOLDER/'execution_release.json',{'scope':'ONLY three named T styles seed42 fourepochs296steps each; no long training',
             'passed':True,'independent_preflight_sha256':sha(FOLDER/'independent_preflight_verification.json'),
             'source_script_sha256':sha(__file__),'styles':list(STYLES),'primary':'LAST296','calibration_or_test_allowed':False,'long_training_authorized':False})
        for style in STYLES:
            cfginfo=report['configs'][style];assert sha(cfginfo['path'])==cfginfo['sha256']
            init=report['initials'][style];assert sha(init['path'])==init['sha256']
            cfg=json.loads(Path(cfginfo['path']).read_text());target=ROOT/cfg['checkpoint_dir']/cfg['run_name']
            assert not target.exists();started=time.perf_counter()
            with (FOLDER/(style+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,'-m','src.train','--config',cfginfo['path'],'--device','cpu'],stdout=log,stderr=subprocess.STDOUT)
                state.update(status='running',style=style,training_pid=child.pid);dump(FOLDER/'progress.json',state)
                if child.wait()!=0:raise RuntimeError(style+' failed; no retry')
            last=torch.load(target/'last.pt',map_location='cpu',weights_only=False);opt=torch.load(target/'optimizer.pt',map_location='cpu',weights_only=False)
            history=json.loads((target/'history.json').read_text())
            assert len(history)==4 and last['extra']['global_step']==opt['global_step']==296
            assert finite_tree(last['state']) and finite_tree(opt) and all(r['val_n']==142 and r['skipped_no_supervision_batches']==0 for r in history)
            dump(FOLDER/(style+'_completion.json'),{'complete':True,'wall_seconds':time.perf_counter()-started,'last_sha256':sha(target/'last.pt'),
                 'optimizer_sha256':sha(target/'optimizer.pt'),'history':history,'independent_quality_evaluation_pending':True})
            state.setdefault('completed_styles',[]).append(style);dump(FOLDER/'progress.json',state)
            del last,opt;gc.collect()
        state.update(status='three_T_arms_complete_pending_independent_quality_evaluation',finished_utc=now());dump(FOLDER/'progress.json',state)
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error));dump(FOLDER/'progress.json',state);raise
    finally:
        (FOLDER/'ACTIVE.lock').unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','verify','run']);args=parser.parse_args()
    {'prepare':prepare,'verify':verify,'run':run}[args.mode]()
