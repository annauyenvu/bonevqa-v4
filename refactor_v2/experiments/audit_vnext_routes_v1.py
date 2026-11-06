"""Aggregate recorded gradient/update evidence without touching running models."""
import hashlib,json,math,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];O=ROOT/'refactor_v2/reports/vnext_local_tokens_v1'
ARMS=('B','real_aux','placebo_aux','grid_aux','real_noaux')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(partial=False):
    available=[a for a in ARMS if (O/f'run_{a}/completion.json').exists()]
    if not partial:assert available==list(ARMS)
    evidence={};sources={}
    selectors={'visual_lora':lambda k:k.startswith('visual.') and 'lora' in k.lower(),'fracture_head':lambda k:k.startswith('fracture_head.'),'local_attention':lambda k:k.startswith('local_tokens.attention.'),'local_question':lambda k:k.startswith('local_tokens.question.'),'local_gate':lambda k:k.startswith('local_tokens.gate.'),'local_auxiliary_only':lambda k:k.startswith(('local_tokens.mil_score.','local_tokens.aux_head.'))}
    for arm in available:
        path=O/f'run_{arm}/completion.json';quality=read(path);gate=O/f'gate_{arm}/completion.json';smoke=read(gate)
        sources[str(path)]=sha(path);sources[str(gate)]=sha(gate)
        assert quality['updates']==292
        g=quality['gradients'];u=quality['parameter_changes'];assert set(g)==set(u)
        assert all(math.isfinite(v) and v>=0 for v in list(g.values())+list(u.values()))
        groups={}
        for name,select in selectors.items():
            keys=[k for k in g if select(k)]
            if not keys:continue
            groups[name]={'tensors_recorded':len(keys),'max_abs_gradient':max(g[k] for k in keys),'max_abs_parameter_change':max(u[k] for k in keys),'tensors_with_nonzero_gradient':sum(g[k]>0 for k in keys)}
        for name in ('visual_lora','fracture_head'):
            assert groups[name]['max_abs_gradient']>0 and groups[name]['max_abs_parameter_change']>0
        if arm!='B':
            for name in ('local_attention','local_question','local_gate'):
                assert groups[name]['max_abs_gradient']>0 and groups[name]['max_abs_parameter_change']>0
            assert any(v>0 for v in smoke['answer_route_gradients'].values())
            assert smoke['mechanics']['fracture_logit_intervention_max']>0
            if arm.endswith('_aux'):assert groups['local_auxiliary_only']['max_abs_gradient']>0
            else:assert groups['local_auxiliary_only']['max_abs_gradient']==0
        evidence[arm]={'groups':groups,'smoke_isolated_answer_route':smoke['answer_route_gradients'],'smoke_mechanics':smoke['mechanics'],'epoch_training_seconds':[h['train_seconds'] for h in quality['history']],'epoch_evaluation_seconds':[h['eval_seconds'] for h in quality['history']],'max_gradients_are_total_training_objective':True}
    result={'partial':partial,'available_arms':available,'evidence':evidence,'dependencies':sources,'implementation_limits':['The local subclass evaluates original core fusion and then local fusion. This consumes additional training dropout draws versus B; spatial controls share this implementation. Equal updates do not imply equal FLOPs or stochastic paths.','For real_noaux, zero auxiliary-only gradients may coexist with nonzero AdamW weight-decay parameter changes. Such changes are not evidence that auxiliary semantics were learned.','Observed nonzero gradients, updates and intervention responses establish technical use, not semantic benefit or clinical generalization.'],'source_sha256':sha(Path(__file__))}
    name='gradient_route_audit_final.json' if not partial else f'gradient_route_audit_partial_{len(available)}arms.json'
    with (O/name).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'artifact':name,'arms_verified':available,'finite_gradients_updates_and_expected_routes':True}))
if __name__=='__main__':main('--partial' in sys.argv)
