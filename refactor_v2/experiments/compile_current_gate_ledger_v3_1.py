"""Evidence crosswalk and fixed box-presence diagnostic, no fitted model."""
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path.cwd();folder=root/'refactor_v2/reports';output=folder/'CURRENT_GATE_LEDGER_V3_1.json'
    assert not output.exists()
    def read(name):return json.loads((folder/name).read_text())
    strength=read('vl_bounded_quality_v3_1/trained_V_strength_TRAIN_only.json')
    assert strength['optimizer_updates']==0 and all(r['weights_exact_and_unchanged'] for r in strength['results'].values())
    rows=read('V_full321_development_v3_1/V_real.json')['records']
    proposals=read('V_full321_proposals_v3_1/boxes.json')
    assert len(rows)==321 and {r['image_id'] for r in rows}==set(proposals)
    counts={k:0 for k in ('TP','FN','TN','FP')}
    for row in rows:
        yes=row['gt'].strip().lower()=='yes';has=bool(proposals[row['image_id']])
        counts[('TP' if has else 'FN') if yes else ('FP' if has else 'TN')]+=1
    assert counts['TP']+counts['FN']==57 and counts['TN']+counts['FP']==264
    sens=counts['TP']/57;spec=counts['TN']/264
    provenance=['data_inventory_exact_v3.json','perceptual_pixel_summary_v3.json',
                'checkpoint_provenance_v3.json','vl_bounded_quality_v3_1/completed_pilot_set_audit.json',
                'V_full321_development_v3_1/independent_artifact_verification.json',
                'L_full321_development_v3_1/independent_artifact_verification.json',
                'vl_bounded_quality_v3_1/trained_V_strength_TRAIN_only.json',
                'detector640_top3_binding_v3/report.json',
                'STEP3_TECHNICAL_REFERENCE_SEAL_V3.json','PROMPT_INTEGRATION_AND_CAUSAL_GATE_V3.md']
    report={'scope':'current v3 evidence ledger; historical reports are not current release authority',
            'source_script_sha256':sha(__file__),'evidence_sha256':{p:sha(folder/p) for p in provenance},
            'fixed_box_presence_ONLY_no_fitting':{'images':321,'positive_images':57,'negative_images':264,
                'confusion_if_any_box_means_yes':counts,'sensitivity':sens,'specificity':spec,
                'binary_score_AUROC':.5*(sens+spec),'balanced_accuracy':.5*(sens+spec),
                'meaning':'proposal existence carries label information without lesion-pixel interpretation',
                'limitations':'same inspected modelval, exploratory diagnostic; not a TRAIN-fitted geometry-only matched model or independent evidence'},
            'diagnostic_findings':[
                'V route is live on four fixed TRAIN QA: nonzero answer gradients and gate-zero logit effects; no evidence of a completely disconnected branch in this diagnostic',
                'Vreal learned gate0.004113; relative visual-stream perturbation4.50e-5 on two QA sharing one automatic-ROI image; zero on empty-ROI rows',
                'No public-answer changes among four TRAIN QA under gate-zero; small effect is observed, but is not established as the cause of weak generalization',
                'At this evaluation-mode TRAIN probe, selected V answer-gradient maxima exceed other-objective maxima; do not claim auxiliary-loss domination or remove contrastive loss without separate evidence',
                '124of321 modelval images have proposals;197empty includes some positives, so automatic localization coverage limits how often semantic local information is available',
                'Existing fixed top3 detector audit localizes26of57positive images atIoU>=0.5, not clinical VQA sensitivity'],
            'gates':{
                'data_and_initialization':{'status':'development seals and exact inventory exist, not untouched confirmation',
                    'remaining':'perceptual screen is candidate-based, not an exhaustive guarantee; preserve PMC train_lon contamination warning and exact checkpoint chains'},
                'CPU_wiring_and_replay':{'status':'tested scopes passed, 5boundedV_L LAST296 and full321 development artifacts verified',
                    'remaining':'do not extend baseline tiny CPU epoch-boundary resume certification to all module/phase/mid-epoch/runtime paths'},
                'V_semantic':{'status':'not passed; location-control AUROC/AP intervals include zero, overall142QA interval includes zero',
                    'remaining':'no-ROI-capacity, image-group shuffled and TRAIN-fitted geometry-only new-fold controls not all run; absence is not permission to expand a currently weak candidate'},
                'L_semantic':{'status':'not passed; real lens below matched original/location QA controls, full321 discriminative intervals include zero',
                    'remaining':'DCL/InfoNCE and pixel-(un)shuffle contribution ablations not established; do not claim exact-author reproduction or Lens benefit'},
                'P_semantic':{'status':'previous GM+LF matched screen negative; PF absent',
                    'remaining':'target-consistency on/off and further contexts not all screened; no P expansion justified by current evidence'},
                'T_semantic':{'status':'technical prompt/token controls only, no current quality release',
                    'remaining':'matched raw/clinical/irrelevant/paraphrase quality experiment on the approved current initialization/data remains unverified'},
                'confirmation_and_calibration':{'status':'not passed',
                    'remaining':'honest component-grouped nested evaluation or verified independent holdout, then matched seeds only for supported candidates; reserved-role calibration not used to fix current development scores'},
                'Final_Solution':{'status':'not selected',
                    'remaining':'B remains provisional reference; historical paper FinalSolution/old step3_gate overfit results do not establish current v3 clinical efficacy'}},
            'next_safe_work':['Review a bounded TRAIN-fitted geometry-only diagnostic under exact current source/role binding before additional multimodal training',
                              'Predeclare candidate architectural/loss changes only if diagnostics support a mechanism; preserve failed candidates and forbid posthoc threshold/gate amplification',
                              'Keep remaining experiment/confirmation obligations explicit; do not label all of step3 complete because the five-arm pilot set finished'],
            'optimizer_updates':0,'calibration_or_test_inference_used':False,'long_training_authorized':False,
            'clinical_root_cause_proven':False,'step3_complete':False,'production_released':False}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['fixed_box_presence_ONLY_no_fitting'],indent=2))


if __name__=='__main__':main()
