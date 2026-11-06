"""Verify four gates and save compact PRE-update state/configuration, no training."""
import json
import shutil
from pathlib import Path
import torch
from experiments.gate_pf_comparison_v2 import initialize, ROOT as GATES
from experiments.pf_comparison_v1 import ARMS
from experiments.lapa_integrated_gate_v1 import REPORTS, digest
from experiments.prepare_lapa_pilot_v1 import sha

OUT=REPORTS/'pf_comparison_preparation_v1'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(8)
    model=initialize('author_real');state=model.state_dict();initial=digest(state.items())
    receipts={arm:json.loads((GATES/arm/'receipt.json').read_text()) for arm in ARMS}
    assert all(r['initial_digest']==initial and r['updates']==2 and r['fixed_state_unchanged'] and
               all(r['changed_groups'].values()) and all(r['replay'].values()) for r in receipts.values())
    assert all(r['microbatches']==receipts[ARMS[0]]['microbatches'] for r in receipts.values())
    names={n for n,p in model.named_parameters(remove_duplicate=False) if p.requires_grad}
    needed=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
    assert shutil.disk_usage(OUT).free-needed>8*1024**3
    torch.save({'trainable':{n:t for n,t in state.items() if n in names},'initial_digest':initial,
                'fixed_digest':digest((n,t) for n,t in state.items() if n not in names),'updates':0,'seed':42},OUT/'shared_PREupdate.pt')
    schedules=REPORTS/'step4_localization_v3_1/schedules.json'
    schedule=json.loads(schedules.read_text())
    assert len(schedule)==4 and all(sum(map(len,e))==1164 for e in schedule)
    assert sum((len(e)+3)//4 for e in schedule)==292
    configs={arm:{'arm':arm,'seed':42,'epochs':4,'updates':292,'batch':4,'accumulation':4,
                  'lr':1e-4,'weight_decay':.01,'clip':1.,'eta':.1,'checkpoint':'LAST4',
                  'initialization':str(OUT/'shared_PREupdate.pt'),'schedule':str(schedules),
                  'quality_released':False} for arm in ARMS}
    (OUT/'configs.json').write_text(json.dumps(configs,indent=2))
    (OUT/'gate_summary.json').write_text(json.dumps({'all_four_passed':True,'same_initial_state':initial,
          'same_microbatches':True,'compact_alias_aware_replay':True,'receipts':receipts},indent=2))
    dependencies=json.loads((REPORTS/'lapa_shared_quality_v1/execution_release.json').read_text())['dependencies']
    for path,h in dependencies.items():assert sha(Path(path))==h,path
    extra=[Path(__file__),OUT/'shared_PREupdate.pt',OUT/'configs.json',OUT/'gate_summary.json',
           REPORTS/'PF_COMPARISON_PROTOCOL_V1.md',schedules,
           REPORTS/'pf_slake_graph_v1/graph_seal.json']
    extra += [Path(p) for p in json.loads((REPORTS/'pf_slake_graph_v1/graph_seal.json').read_text())['artifacts']]
    experiments=Path(__file__).parent
    extra += [experiments/n for n in ['pf_comparison_v1.py','gate_pf_comparison_v2.py','pf_reconstructed_v2.py',
               'pf_masked_variant_v1.py','pf_slake_integrated_v1.py','pf_slake_reference_v1.py']]
    for arm in ARMS:extra += [GATES/arm/'receipt.json',GATES/arm/'replay.json']
    dependencies.update({str(p):sha(p) for p in extra})
    (OUT/'preparation_seal.json').write_text(json.dumps({'dependencies':dependencies,'quality_released':False,
         'pending':['full evaluator check for new runner','versioned quality worker/release','quality disk feasibility']},indent=2))
    print(json.dumps({'gates':4,'initial_digest':initial,'dependency_count':len(dependencies),
                      'free_bytes':shutil.disk_usage(OUT).free,'quality_released':False}))


if __name__=='__main__':main()
