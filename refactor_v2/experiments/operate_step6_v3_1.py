"""Wait for audited screening, then gated five-seed training and fixed assessment."""
import os,sys,time,subprocess,gc
from pathlib import Path
from experiments.prepare_step4_localization_v3_1 import OUT as PARENT,ROOT,R,read,dump
from experiments.vl_bounded_proposal_contract_v3_1 import sha
OUT=R/'step6_confirmation_v3_1';SCREEN=R/'step56_screening_v3_1'
WORKER=ROOT/'refactor_v2/experiments/step6_seed_worker_v3_1.py'
ASSESS=ROOT/'refactor_v2/experiments/step6_assessment_v3_1.py'

def invoke(args,log):
    with log.open('x') as f:subprocess.run([sys.executable,'-u',*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)

def verify_gate(seed,arm):
    import torch
    from src.models.bonevqa import BoneVQAModel
    from experiments.step6_assessment_v3_1 import predict
    torch.set_num_threads(8);folder=OUT/f'seed{seed}_{arm}';target=folder/'smoke';done=read(target/'completion.json')
    assert done['disposable'] and done['updates']==(4 if arm.startswith('L') else 2)
    family='latent.' if arm.startswith('P') else 'lens_fusion.' if arm.startswith('L') else 'fracture_head.'
    assert any(g>0 and done['parameter_updates'][k]>0 for k,g in done['gradients'].items() if k.startswith(family))
    epoch=len(done['history']);checkpoint=target/f'epoch{epoch}.pt';digest=sha(checkpoint)
    model=BoneVQAModel.load(str(checkpoint),device='cpu').eval()
    actual=predict(model,read(PARENT/'model_val.json')[:8]);assert actual==read(target/f'epoch{epoch}_evaluation.json')['records'] and sha(checkpoint)==digest
    dump(target/'fresh_replay.json',{'passed':True,'outputs_exact':True,'checkpoint_sha256':digest})

def operate():
    assert not (OUT/'operation_started.json').exists();dump(OUT/'operation_started.json',{'pid':os.getpid(),'source_sha256':sha(__file__),'waiting_for_audited_screening':True})
    try:
        while not (SCREEN/'screening_selection.json').exists():
            if any((SCREEN/n).exists() for n in ('operation_failure.json','postprocessing_failure.json')):raise RuntimeError('Screening failed; inspect before continuation')
            time.sleep(30)
        selection=read(SCREEN/'screening_selection.json');assert selection['passed_metric_budget_replay_checks']
        for p,h in selection['dependencies'].items():assert sha(p)==h,p
        arms=selection['step6_arms'];assert arms[0]=='B' and all(not a.startswith('V') for a in arms)
        dump(OUT/'selected_arms.json',{'arms':arms,'seeds':[42,43,44,45,46],'selection_sha256':sha(SCREEN/'screening_selection.json'),'no_module_promoted':not selection['selected']})
        for seed in (43,44,45,46):invoke([WORKER,'initialize',seed,*arms],OUT/f'seed{seed}_initialize.log')
        deps=list(OUT.glob('*.json'))+list(OUT.glob('seed*_*/*.json'))+list(OUT.glob('seed*_*/*.pt'))
        deps += [WORKER,ASSESS,Path(__file__).resolve(),SCREEN/'screening_selection.json',R/'STEP6_LOCKED_DESIGN_V3_1.md']+list((ROOT/'refactor_v2/src').rglob('*.py'))
        deps += [ROOT/'refactor_v2/experiments'/n for n in ('grouped_calibration_v3_1.py','grouped_public_evaluation_core_v3_1.py','prepare_step4_localization_v3_1.py','vl_bounded_proposal_contract_v3_1.py')]
        binding=read(OUT/'role_binding.json')
        dependency={str(p):sha(p) for p in deps};dependency.update(binding['dependencies'])
        # Recheck shared initial tensors within every seed, before any quality update.
        import torch
        for seed in (43,44,45,46):
            baseline=torch.load(OUT/f'seed{seed}_B/PREtask.pt',map_location='cpu',weights_only=False)['state']
            for arm in arms:
                incoming=torch.load(OUT/f'seed{seed}_{arm}/PREtask.pt',map_location='cpu',weights_only=False)['state']
                assert all(torch.equal(v,incoming[k]) for k,v in baseline.items())
                del incoming
            del baseline;gc.collect()
        dump(OUT/'gate_release.json',{'dependencies':dependency,'quality_training_released':False})
        for seed in (43,44,45,46):
            for arm in arms:
                invoke([WORKER,'smoke',seed,arm],OUT/f'seed{seed}_{arm}_smoke.log')
                invoke([Path(__file__).resolve(),'verify_gate',seed,arm],OUT/f'seed{seed}_{arm}_gate_replay.log')
        dump(OUT/'training_release.json',{'dependencies':dependency,'training_released':True,'arms':arms,'seeds_new':[43,44,45,46],'reused_seed42':True,'updates_each':292})
        checkpoints={'42':{}}
        for arm in arms:
            path=PARENT/'run_B/epoch_4.pt' if arm=='B' else SCREEN/('run_'+arm)/'epoch4.pt'
            checkpoints['42'][arm]={'path':str(path),'sha256':sha(path)}
        for seed in (43,44,45,46):
            checkpoints[str(seed)]={}
            for arm in arms:
                invoke([WORKER,'quality',seed,arm],OUT/f'seed{seed}_{arm}_quality.log')
                path=OUT/f'seed{seed}_{arm}/quality/epoch4.pt';done=read(path.parent/'completion.json');assert done['updates']==292
                checkpoints[str(seed)][arm]={'path':str(path),'sha256':sha(path)}
        assessment_deps=dict(dependency)
        for seeds in checkpoints.values():
            for c in seeds.values():assessment_deps[c['path']]=c['sha256']
        dump(OUT/'assessment_release.json',{'dependencies':assessment_deps,'checkpoints':checkpoints,'raw_threshold':.5,'calibrated_threshold':.5,'calibration_only_fit':True,'assessment_scope':'development-informed, not independent clinical confirmation'})
        for seed in (42,43,44,45,46):
            for arm in arms:
                invoke([ASSESS,'measure',seed,arm],OUT/f'assessment_seed{seed}_{arm}.log')
                invoke([ASSESS,'replay',seed,arm],OUT/f'assessment_seed{seed}_{arm}_replay.log')
        dump(OUT/'execution_completed.json',{'arms':arms,'five_training_seeds':[42,43,44,45,46],'all_assessment_fresh_replays_passed':True,'independent_metric_bootstrap_and_report_pending':True,'production_released':False})
    except Exception as error:
        dump(OUT/'operation_failure.json',{'error':repr(error),'preserve_outputs_no_automatic_retry':True});raise
if __name__=='__main__':
    if len(sys.argv)>1:verify_gate(int(sys.argv[2]),sys.argv[3])
    else:operate()
