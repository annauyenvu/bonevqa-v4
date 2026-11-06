"""Preserved-run repair: distinguish threshold metadata from QA metrics.

All scientific choices and inference remain identical to predeclared v3_1.
"""
import argparse
import gc
import json
from pathlib import Path

from experiments import evaluate_T_bounded_current_v3_2 as repaired

core = repaired.core
PREVIOUS_WRAPPER = Path(repaired.__file__).resolve()
core.__file__ = str(Path(__file__).resolve())
core.OUT = core.ROOT/'refactor_v2/reports/T_independent_quality_current_v3_3'


def compare_QA_metrics(metrics, history):
    assert metrics['fracture_yes_threshold']==.5
    expected = {k[4:]:v for k,v in history.items() if k.startswith('val_') and k!='val_loss'}
    actual = {k:v for k,v in metrics.items() if k!='fracture_yes_threshold'}
    assert set(actual)==set(expected), (set(actual)-set(expected),set(expected)-set(actual))
    assert all(abs(actual[k]-expected[k])<=1e-6 for k in expected), {
        k:(actual[k],expected[k]) for k in expected if abs(actual[k]-expected[k])>1e-6}


def evaluate(qa, full, indices, checkpoints, seed):
    for style in core.STYLES:
        info = checkpoints[style]
        core.set_seed(seed)
        model = core.BoneVQAModel.load(info['path'],device='cpu').eval()
        assert model.cfg['textual_prompt_style']==style and not model.cfg['use_region_in_prompt']
        assert model.cfg['fracture_yes_threshold']==.5 and model.cfg.get('fracture_platt_slope') is None
        assert not any(model.cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt'))
        result = {'QA142':core.predict(model,qa,list(range(len(qa)))),
                  'fracture321':core.predict(model,full,indices)}
        compare_QA_metrics(result['QA142']['metrics'],info['history'][-1])
        # Explicit score-threshold decision and saved public answers must agree;
        # only the threshold metadata field is absent from training-log metrics.
        by_answers = core.compute_metrics(result['QA142']['records'])
        assert by_answers=={k:v for k,v in result['QA142']['metrics'].items() if k!='fracture_yes_threshold'}
        if style=='clinical':
            original = model.cfg['textual_prompt_style']
            try:
                model.cfg['textual_prompt_style']='clinical_paraphrase'
                result['secondary_inference_paraphrase142']=core.predict(model,qa,list(range(len(qa))))
            finally:
                model.cfg['textual_prompt_style']=original
        assert core.sha(info['path'])==info['sha256']
        del model
        gc.collect()
        yield style,result


core.evaluate = evaluate


def prepare():
    previous = core.ROOT/'refactor_v2/reports/T_independent_quality_current_v3_2/progress.json'
    failure = json.loads(previous.read_text())
    assert failure['status']=='failed_preserved_no_automatic_retry'
    assert failure['failure']=="KeyError('val_fracture_yes_threshold')"
    repaired.prepare()
    path = core.OUT/'evaluation_plan.json'
    plan = json.loads(path.read_text())
    for dependency in [PREVIOUS_WRAPPER,previous]:
        plan['dependencies_sha256'][str(dependency)] = core.sha(dependency)
    plan['threshold_metadata_comparison_repair'] = {
        'previous_failed_folder':str(previous.parent),
        'reason':'Explicit threshold metadata is not a stored training-log metric',
        'require_threshold':.5,'metric_key_set_must_match_training_history_exactly':True,
        'all_metric_values_must_replay_tolerance':1e-6,
        'public_answer_vs_score_threshold_metrics_must_match':True,
        'original_artifacts_overwritten':False,'scientific_protocol_changed':False}
    core.dump(path,plan)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','run','verify'])
    args = parser.parse_args()
    {'prepare':prepare,'run':core.run,'verify':core.replay}[args.mode]()
