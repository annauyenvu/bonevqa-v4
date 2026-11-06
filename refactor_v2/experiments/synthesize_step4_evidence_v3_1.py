"""Evidence inventory; never equates historical cohorts with new matched pilot."""
from experiments.prepare_step4_localization_v3_1 import OUT,R,read,dump
from experiments.vl_bounded_proposal_contract_v3_1 import sha
def main():
    target=OUT/'historical_LP_evidence.json'; assert not target.exists()
    source=R/'vl_bounded_quality_v3_1/completed_pilot_set_audit.json'
    audit=read(source); assert audit['passed_artifact_and_replay_audit']
    L={}
    for arm in ('L_real','L_original_teacher','L_location_placebo'):
        folder=source.parent/arm; verify=folder/'independent_LAST296_verification.json'
        assert sha(verify)==audit['results'][arm]['independent_report_sha256']
        assert sha(folder/'last.pt')==audit['results'][arm]['checkpoint_sha256']
        L[arm]={'verified':True,'history':read(folder/'history.json'),'LAST296_metrics':audit['results'][arm]['metrics'],'sources':{str(p):sha(p) for p in (verify,folder/'history.json')}}
    P={}
    for arm in ('P','P_random_bank','P_unconditioned'):
        path=R/'P_target_semantic_pilot_v3'/(arm+'_seed42_target_bounded_v3_completion.json')
        completed=read(path); assert completed['complete'] and len(completed['history'])==4
        P[arm]={'history':completed['history'],'wall_seconds':completed['wall_seconds'],'source':str(path),'sha256':sha(path)}
    interpretation=R/'P_target_semantic_pilot_v3/interpretation.json'
    dump(target,{'historical_L':L,'historical_P':P,'P_interpretation':read(interpretation),'P_interpretation_sha256':sha(interpretation),'scope':'Historical seed42 pilots:261TRAIN images1173QA;32-image142QA learning curves plus321-image canonical re-evaluation. New pilot uses256TRAIN images1164QA and full1432QA model-validation. Do not pool or rank across these cohorts as matched comparisons.','step4_complete':False,'remaining':'New B/V pilot actual quality results, fresh output replay and paired analysis; evaluate historical L/P coverage against explicit exit criteria before promoting to screening.'})
    print('Historical L/P curves and controls inventoried with hashes')
if __name__=='__main__': main()
