"""Deferred curves, independent confirmation audit and report closure."""
import os,sys,time,subprocess
from experiments.prepare_step4_localization_v3_1 import ROOT,R,OUT as PILOT,read,dump
from experiments.vl_bounded_proposal_contract_v3_1 import sha
SCREEN=R/'step56_screening_v3_1';CONFIRM=R/'step6_confirmation_v3_1'
def wait_for(path,failures):
    while not path.exists():
        if any(p.exists() for p in failures):raise RuntimeError('Upstream failure preserved; inspect logs')
        time.sleep(30)
def screening():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=read(SCREEN/'screening_selection.json');assert s['passed_metric_budget_replay_checks']
    histories=dict(s['history'])
    for a in ('B','V_real','V_placebo'):histories[a]=read(PILOT/('run_'+a)/'completion.json')['history'][1:]
    fig,axes=plt.subplots(3,2,figsize=(13,12))
    for group,axrow in zip((('B','V_real','V_placebo'),('B','P_real','P_random','P_unconditioned'),('L_real','L_original','L_placebo')),axes):
        for arm in group:
            h=histories[arm];axrow[0].plot([r['epoch'] for r in h],[r.get('loss',r.get('mean_train_loss')) for r in h],marker='o',label=arm)
            scored=[r for r in h if r.get('metrics') is not None]
            axrow[1].plot([r['epoch'] for r in scored],[r['metrics']['fracture_auprc'] for r in scored],marker='o',label=arm)
        for ax in axrow:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=7)
        axrow[0].set_title('TRAIN loss; phase-specific for L');axrow[1].set_title('Model-validation fracture AP; fixed LAST4 selection')
    fig.tight_layout();fig.savefig(SCREEN/'screening_learning_curves.png',dpi=180);plt.close(fig)
    rows=['# Step5 common-cohort screening outcome','', '| Arm | AP LAST |','|---|---:|']
    rows += [f'| {a} | {v:.6f} |' for a,v in s['AP'].items()]
    rows += ['',f"Selected: {s['selected'] or 'no module; baseline stability only'}.",'','The rule was frozen before new P/L quality outcomes: candidate exceeds baseline and each own matched control AP by0.005. This is a budget/selection rule, not significance. B/V seed42 outputs are inspected pilot reuse. L phases are146alignment+146VQAupdates and have their own same-phase controls; do not claim equal-phase baseline ablation. All arms use256TRAIN1164QA/full321modelval1432QA. No independent clinical validation, no production selection.','', 'Fresh-process full replay and point AP/AUROC checks passed for six new arms. Bootstrap uncertainty for screening selection remains descriptive and is not used to change the frozen rule.']
    with (SCREEN/'SCREENING_OUTCOME.md').open('x') as f:f.write('\n'.join(rows)+'\n')
def confirmation():
    analysis=read(CONFIRM/'independent_analysis.json');assert analysis['passed']
    rows=['# Five-seed development-informed confirmation','', 'Seeds42–46; same assessment images repeated across seeds, not stacked as independent cases. Calibration fits only designated calibration rows, fixed raw/calibrated0.5. Known components are not verified patients and assessment was previously inspected. No untouched independent clinical confirmation or production readiness claim.','', '| Arm/variant | Mean AP ± sample SD | Sensitivity mean | Specificity mean | BA mean |','|---|---:|---:|---:|---:|']
    for name,m in analysis['five_seed_aggregate'].items():
        if m:rows.append(f"| {name} | {m['AP']['mean']:.5f} ± {m['AP']['sample_SD']:.5f} | {m['sensitivity']['mean']:.5f} | {m['specificity']['mean']:.5f} | {m['BA']['mean']:.5f} |")
    rows += ['', 'See independent_analysis.json for every seed, confusion counts, calibration checks and paired component intervals. Intervals hold the five fitted models fixed and do not cover all possible training seeds or unknown patient relationships. Negative or failed calibration is retained without threshold search. A completed run is not proof of architectural benefit.','', 'Screening and repeated-evaluation execution complete within the frozen scopes. Full original seven-step/Final Solution/production certification remains outside this result.']
    with (CONFIRM/'FIVE_SEED_RESULTS.md').open('x') as f:f.write('\n'.join(rows)+'\n')
    paths=[SCREEN/'screening_selection.json',SCREEN/'SCREENING_OUTCOME.md',SCREEN/'screening_learning_curves.png',CONFIRM/'independent_analysis.json',CONFIRM/'FIVE_SEED_RESULTS.md',CONFIRM/'training_release.json',CONFIRM/'assessment_release.json']
    paths+=list(CONFIRM.glob('assessment_seed*.json'))
    dump(CONFIRM/'steps56_closure.json',{'execution_and_analysis_complete':True,'scope':'Fixed common-cohort screening and five-seed development-informed assessment; no independent confirmation','artifacts':{str(p):sha(p) for p in paths},'production_released':False,'full_seven_step_complete':False})
def main():
    assert not (CONFIRM/'report_supervisor_started.json').exists();dump(CONFIRM/'report_supervisor_started.json',{'pid':os.getpid(),'source_sha256':sha(__file__)})
    try:
        wait_for(SCREEN/'screening_selection.json',[SCREEN/'operation_failure.json',SCREEN/'postprocessing_failure.json'])
        screening()
        wait_for(CONFIRM/'execution_completed.json',[CONFIRM/'operation_failure.json'])
        with (CONFIRM/'independent_analysis.log').open('x') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'refactor_v2/experiments/audit_step6_confirmation_v3_1.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
        confirmation()
    except Exception as error:
        dump(CONFIRM/'report_failure.json',{'error':repr(error),'preserve_outputs':True});raise
if __name__=='__main__':main()
