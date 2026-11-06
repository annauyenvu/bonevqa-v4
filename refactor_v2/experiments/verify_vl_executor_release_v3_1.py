"""Fresh-process sealed-input and optimizer-horizon check; no model training."""
import json
import math
from pathlib import Path
import torch
from experiments.vl_bounded_proposal_contract_v3_1 import sha
from experiments.run_vl_bounded_pilot_v3_1 import assert_update_budget, check_environment


def main():
    check_environment()
    root=Path.cwd()
    folder=root/'refactor_v2/reports/vl_bounded_quality_v3_1'
    destination=folder/'independent_execution_verification.json'
    assert not destination.exists()
    release=json.loads((folder/'execution_release.json').read_text())
    assert release['passed'] and release['total_updates_per_arm']==296
    assert not release['long_training_released'] and not release['calibration_or_test_allowed']
    for path,digest in release['dependency_sha256'].items():
        assert sha(path)==digest
    for row in release['fresh_initializations'].values():
        assert sha(row['path'])==row['sha256']
    observations={}
    # Scalar optimizer tests the real tail flush and phase horizon without
    # any patient image, neural-model update, or reusable learned weight.
    for epochs in (2,4):
        cap=epochs*74
        scalar=torch.nn.Parameter(torch.tensor(1.))
        opt=torch.optim.AdamW([scalar],lr=1e-4,betas=(.9,.98),weight_decay=.01)
        warmup=int(.05*cap)
        def multiplier(step):
            if step<warmup:return (step+1)/max(1,warmup)
            return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,cap-warmup))))
        sched=torch.optim.lr_scheduler.LambdaLR(opt,multiplier)
        cumulative=0;steps_by_epoch=[];tail_grad=[]
        for epoch in range(epochs):
            updates=0;opt.zero_grad(set_to_none=True)
            for it in range(294):
                (scalar/4).backward()
                if (it+1)%4==0 or it+1==294:
                    assert_update_budget(cumulative,cumulative,cap)
                    if it+1==294:tail_grad.append(float(scalar.grad))
                    opt.step();opt.zero_grad(set_to_none=True);sched.step()
                    updates+=1;cumulative+=1
            assert updates==74
            steps_by_epoch.append(updates)
        assert cumulative==cap and sched.last_epoch==cap and sched.get_last_lr()[0]==0.
        assert tail_grad==[.5]*epochs
        try:assert_update_budget(cap,cap,cap)
        except RuntimeError:pass
        else:raise AssertionError('Extra update accepted')
        observations[str(epochs)]={'updates':cumulative,'updates_per_epoch':steps_by_epoch,
                                  'tail_2_microbatches_keep_divisor4':tail_grad,'final_lr':0.,
                                  'extra_update_rejected':True,'scalar_test_weights_discarded':True}
    destination.write_text(json.dumps({'passed':True,'scope':'sealedfile_and_scalar_optimizer_verification_ONLY',
        'execution_release_sha256':sha(folder/'execution_release.json'),
        'sealed_dependency_count':len(release['dependency_sha256']),
        'fresh_initializations_verified':5,'optimizer_horizons':observations,
        'patient_model_updates':0,'calibration_or_test_used':False},indent=2)+'\n')
    print('Independent execution seal and 148/296-step scalar horizons verified; zero patient-model updates')


if __name__=='__main__':main()
