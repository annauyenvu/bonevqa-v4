"""Four fixed TRAIN QA learned-route diagnostic; zero optimizer updates.

This can show a live/small route, not generalization or a causal explanation
for weak development results. No labels/regions are passed to public inference.
"""
import gc
import hashlib
import json
from pathlib import Path

import torch
from src.data import build_dataset,collate_fn
from src.models.bonevqa import BoneVQAModel
from src.train import build_split
from src.utils import set_seed
from experiments.run_vl_bounded_pilot_v3_1 import check_environment
from experiments.vl_bounded_proposal_contract_v3_1 import sha


def hashes(model):return {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}


def main():
    check_environment();root=Path.cwd();parent=root/'refactor_v2/reports/vl_bounded_quality_v3_1'
    output=parent/'trained_V_strength_TRAIN_only.json';assert not output.exists()
    assert not (parent/'ACTIVE.lock').exists()
    audited=json.loads((parent/'completed_pilot_set_audit.json').read_text())
    assert audited['passed_artifact_and_replay_audit']
    base=json.loads((root/'refactor_v2/reports/fracatlas_scale_pilot_v3/fracatlas_B_roco_scale_pilot_v3.json').read_text())
    kwargs=dict(base['data']['loader_kwargs'])
    kwargs.update(with_masks=False,with_boxes=False,boxes_from=str(root/'refactor_v2/reports/vl_pilot_proposals640_v3_1/boxes.json'),boxes_as_masks=True,roi_control='real')
    ds=build_split(build_dataset,base['data']['dataset'],'train',None,True,kwargs)
    control_root=root/'refactor_v2/reports/vl_matched_control_audit_v3_1'
    audit=json.loads((control_root/'report.json').read_text())
    indices=audit['selected_TRAIN_QA_indices']
    batch=collate_fn([ds[i] for i in indices]);assert len(indices)==4
    geometry=json.loads((control_root/'group_translation_location_placebo.json').read_text())
    results={}
    for arm in ('V_real','V_location_placebo'):
        checkpoint=parent/arm/'last.pt'
        assert sha(checkpoint)==audited['results'][arm]['checkpoint_sha256']
        set_seed(90210);model=BoneVQAModel.load(str(checkpoint),device='cpu').eval()
        before=hashes(model)
        boxes=batch['boxes'] if arm=='V_real' else [geometry[i] or None for i in batch['image_ids']]
        masks=model._prepare_masks(batch['images'],[None]*4,boxes,batch['image_ids'])
        roi=[len(m)>0 for m in masks]
        @torch.no_grad()
        def endpoints(off):
            model.cfg['visual_force_zero']=off
            core=model._encode_core(batch['images'],batch['questions'],[None]*4,masks,model.cfg['prompt_kind'],False)
            delta=core['vis']['f_i']-core['vis']['global_tokens']
            relative=delta.norm(dim=(1,2))/core['vis']['global_tokens'].norm(dim=(1,2)).clamp_min(1e-12)
            result={'closed':core['logits_closed'].detach(),'fracture':core['fracture_logit'].detach(),
                    'relative_visual_stream_delta':relative.tolist()}
            opened=[i for i,t in enumerate(batch['answer_types']) if t=='OPEN']
            if opened:
                prefix=model._llm_prefix(core)[opened]
                embeds,attention,labels=model._build_llm_inputs(prefix,[core['q_texts'][i] for i in opened],[batch['answers'][i] for i in opened])
                logits=model.llm(inputs_embeds=embeds,attention_mask=attention).logits
                result['open_teacherforced']=logits[:,:-1][labels[:,1:]!=-100].detach().float()
            public={'images':batch['images'],'questions':batch['questions'],'boxes':boxes}
            result['public_answers']=[r['answer'] for r in model.predict_batch(public)]
            return result
        try:
            on=endpoints(False);off=endpoints(True)
            differences={k:float((on[k]-off[k]).abs().max()) for k in ('closed','fracture','open_teacherforced') if k in on}
            empty=[i for i,has in enumerate(roi) if not has]
            empty_errors={k:float((on[k][empty]-off[k][empty]).abs().max()) for k in ('closed','fracture')} if empty else {}
            assert all(v==0 for v in empty_errors.values())
            model.cfg['visual_force_zero']=False
            # Differentiable answer/auxiliary gradient audit, no backward step.
            model_batch=dict(batch);model_batch['boxes']=boxes;model_batch['masks']=masks
            named=dict(model.named_parameters());names=['prompt_mixer.gamma_visual','prompt_mixer.visual_proj.0.weight']
            with torch.enable_grad():
                out=model(model_batch,stage=3)
                grads={}
                for label,loss in (('answer_only',out['loss_answer']),('other_objectives',out['loss']-out['loss_answer'])):
                    values=torch.autograd.grad(loss,[named[k] for k in names],retain_graph=True,allow_unused=True)
                    grads[label]={k:0. if v is None else float(v.abs().max()) for k,v in zip(names,values)}
                    assert all(v is None or torch.isfinite(v).all() for v in values)
            assert hashes(model)==before and sha(checkpoint)==audited['results'][arm]['checkpoint_sha256']
            results[arm]={'checkpoint_sha256':sha(checkpoint),'learned_gate':float(model.prompt_mixer.gamma_visual.detach()),
                          'proposal_present_by_QA':roi,'relative_visual_stream_delta_per_QA':on['relative_visual_stream_delta'],
                          'gate_zero_max_logit_differences':differences,'no_ROI_rows_gate_zero_errors':empty_errors,
                          'public_answer_changes_on_four_TRAIN_QA':sum(a!=b for a,b in zip(on['public_answers'],off['public_answers'])),
                          'selected_gradient_max':grads,'weights_exact_and_unchanged':True}
            del out,values,model,named,on,off;gc.collect()
        finally:
            # The temporary flag is never saved to any checkpoint.
            pass
    report={'scope':'four preselected pilotTRAIN QA only; trained V route strength NOT efficacy',
            'source_script_sha256':sha(__file__),'TRAIN_indices':indices,'results':results,
            'optimizer_updates':0,'calibration_or_test_used':False,'public_input_keys':['images','questions','automatic_boxes_or_declared_control'],
            'teacherforced_answer_targets':'existing TRAIN answers ONLY for logged logits, not public inference',
            'limitations':['Four TRAIN rows not representative quality evidence or proof of failure cause',
                           'Answer vs auxiliary gradient magnitudes are diagnostics, not automatic loss-weight recommendations',
                           'Do not amplify a learned gate or change loss after seeing this without a separately bounded technical plan']}
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
