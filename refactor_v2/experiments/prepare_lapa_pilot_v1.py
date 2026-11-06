"""Save pre-update state and audit existing gates; never train."""
import gc
import hashlib
import json
from pathlib import Path
import torch
from experiments.lapa_pilot_runner_v1 import PilotRunner, ARMS
from experiments.lapa_integrated_gate_v1 import REPORTS, digest

OUT=REPORTS/'lapa_pilot_preparation_v1'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=False); torch.set_num_threads(8);torch.manual_seed(42)
    model=PilotRunner('train_cs1')
    checkpoint=REPORTS/'step4_localization_v3_1/B_PREtask.pt'
    release=json.loads((REPORTS/'vnext_local_tokens_v1/training_release.json').read_text())
    assert sha(checkpoint)==release['dependencies'][str(checkpoint)]
    old=torch.load(checkpoint,map_location='cpu')['state']
    supplied={k[7:]:v for k,v in old.items() if k.startswith('visual.')}
    status=model.visual.load_state_dict(supplied,strict=False);assert not status.unexpected_keys
    assert all(torch.equal(v,model.visual.state_dict()[k]) for k,v in supplied.items())
    del old,supplied
    state=model.state_dict();initial=digest(state.items())
    summaries={}
    prefixes={'GM':'reference.prompt.generation.','LF':'reference.prompt.LF.',
              'text_MM':'reference.text_layers.','image_MM':'reference.image_layers.',
              'head':'head.','PF':'reference.prompt.PF.'}
    for arm in ARMS:
        gate=REPORTS/'lapa_pilot_gates_v1'/arm
        receipt=json.loads((gate/'receipt.json').read_text());assert receipt['initial_state_digest']==initial
        after=torch.load(gate/'state.pt',map_location='cpu')['state']
        summaries[arm]={}
        for group,prefix in prefixes.items():
            keys=[k for k in state if k.startswith(prefix)]
            changed=[k for k in keys if not torch.equal(state[k],after[k])]
            maximum=max((after[k]-state[k]).abs().max().item() for k in keys)
            summaries[arm][group]={'tensors':len(keys),'changed':len(changed),'max_abs_delta':maximum}
            assert (not changed) if group=='PF' or (arm=='carrier' and group in ('GM','LF')) else bool(changed),(arm,group)
        del after;gc.collect()
    torch.save({'state':state,'seed':42,'updates':0,'state_digest':initial},OUT/'shared_PREupdate.pt')
    schedule_path=REPORTS/'step4_localization_v3_1/schedules.json'
    schedules=json.loads(schedule_path.read_text())
    assert len(schedules)==4 and all(sum(len(b) for b in epoch)==1164 for epoch in schedules)
    updates=sum((len(epoch)+3)//4 for epoch in schedules);assert updates==292
    configs={arm:{'arm':arm,'seed':42,'epochs':4,'updates':292,'batch_size':4,'accumulation':4,
                  'lr':1e-4,'weight_decay':.01,'clip':1.,'checkpoint_rule':'LAST4',
                  'init':str(OUT/'shared_PREupdate.pt'),'schedule':str(schedule_path),
                  'eta':values[1],'quality_released':False} for arm,values in ARMS.items()}
    with (OUT/'configs.json').open('x') as f:json.dump(configs,f,indent=2)
    with (OUT/'module_update_audit.json').open('x') as f:json.dump(summaries,f,indent=2)
    deps=[OUT/'shared_PREupdate.pt',OUT/'configs.json',OUT/'module_update_audit.json',schedule_path,
          REPORTS/'LAPA_SHARED_REFERENCE_PILOT_PROTOCOL_V1.md',Path(__file__)]
    with (OUT/'preparation_seal.json').open('x') as f:json.dump({'initial_state_digest':initial,'quality_released':False,'hashes':{str(p):sha(p) for p in deps}},f,indent=2)
    print(json.dumps({'initial_state_digest':initial,'all_module_audits_passed':True,'updates':updates,'quality_released':False}))


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(e)},f)
        raise
