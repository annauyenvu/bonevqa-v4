"""Audit recorded gradient summaries/state hashes; no neural imports/models."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/fullfold0_fracture_gradient_probe_v3_5'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


if __name__=='__main__':
    destination=OUT/'independent_artifact_audit.json';assert not destination.exists()
    plan=read(OUT/'protocol.json');report=read(OUT/'report.json')
    assert report['protocol_sha256']==sha(OUT/'protocol.json')
    assert read(OUT/'progress.json')['report_sha256']==sha(OUT/'report.json')
    assert report['results_sha256']==sha(OUT/'results.json') and report['replay_sha256']==sha(OUT/'fresh_process_replay.json')
    replay=read(OUT/'fresh_process_replay.json')
    assert replay['passed'] and replay['all_values_gradients_public_probabilities_exact']
    assert replay['optimizer_updates']==report['optimizer_updates']==0
    for path,digest in plan['dependencies_sha256'].items():assert sha(path)==digest,path
    assert read(OUT/'independent_preflight.json')['protocol_sha256']==sha(OUT/'protocol.json')
    states=read(OUT/'state_snapshots.json')
    assert len(states)==10
    assert all(s['keys']==263 and not s['added_keys'] and not s['removed_keys'] and not s['changed_value_hashes'] for s in states)
    results=read(OUT/'results.json');summaries={}
    rows=read(plan['TRAIN_manifest']);labels=[rows[i]['answer']=='yes' for i in plan['selected_indices']]
    for mode,result in results.items():
        assert sha(plan['targets'][mode]['path'])==result['checkpoint_sha256']==plan['targets'][mode]['sha256']
        assert result['public_sigmoid_exact'] and result['all263weights_unchanged'] and result['same_raw0_5_cross_mode_decisions']
        assert all(v==0 for v in result['head_total_vs_manualBCE_gradient_max_errors'].values())
        assert result['cross_mode_max_abs_error']==max(abs(a-b) for a,b in zip(result['grad_enabled_probabilities'],result['public_probabilities']))
        for actual,prob,target in zip(result['BCE_logit_gradient'],result['grad_enabled_probabilities'],labels):
            expected=(prob-float(target))*(2110/458 if target else 1)/4
            assert abs(actual-expected)<=1e-7+1e-6*abs(expected)
            assert actual<0 if target else actual>0
        for parameter,values in result['gradients'].items():
            for loss,value in values.items():
                if value['connected']:
                    assert all(math.isfinite(value[k]) and value[k]>=0 for k in ('norm','max_abs'))
                else:assert value['norm'] is None and value['max_abs'] is None
        for name in result['head_total_vs_manualBCE_gradient_max_errors']:
            g=result['gradients'][name]
            assert g['BCE']==g['total'] and not g['CE']['connected']
            assert g['BCE']['norm']>0
        assert all(not g['connected'] for g in result['gradients']['qformer.query'].values())
        assert result['gradients']['shared_x_f']['BCE']['norm']>0
        assert result['gradients']['shared_x_f']['CE']['norm']>0
        assert 0<result['BCE_CE_cosines']['shared_x_f']<=1
        summaries[mode]={'shared_BCE_CE_cosine':result['BCE_CE_cosines']['shared_x_f'],
            'cross_mode_max_abs_error':result['cross_mode_max_abs_error'],
            'fracture_head_BCE_total_max_error':max(result['head_total_vs_manualBCE_gradient_max_errors'].values()),
            'QFormer_query_disconnected_on_this_allCLOSED_batch':True}
    result={'passed':True,'report_sha256':sha(OUT/'report.json'),'auditor_sha256':sha(__file__),
        'recorded_gradient_summaries_logit_gradient_public_probabilities_replayed_and_bound':True,
        'limits':'Fresh replay compares saved summaries/logit gradients, not all unpersisted full gradient arrays; four eval-mode TRAIN examples, not stochastic TRAIN/full-dataset gradient coverage',
        'all263state_hashes_unchanged_all10snapshots':True,'physical_LAST_hashes_unchanged':True,
        'observed':summaries,'neural_models':0,'optimizer_updates':0,'new_quality_training_released':False}
    destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
