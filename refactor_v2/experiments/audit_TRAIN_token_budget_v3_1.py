"""Offline tokenizer-only audit, TRAIN labels unchanged, no model inference."""
import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from transformers.utils.hub import cached_file
from src.models.bonevqa import BoneVQAModel
from src.utils import normalize_answer
from experiments.vl_bounded_proposal_contract_v3_1 import sha

ROOT=Path.cwd();OUT=ROOT/'refactor_v2/reports/TRAIN_token_budget_v3_1.json'


def main():
    assert not OUT.exists(),'Preserve prior audit'
    folder=ROOT/'refactor_v2/reports/B_sequence_loss_pilots_v3_3'
    configs={m:json.loads((folder/(m+'.json')).read_text()) for m in ('sequence_uniform','sequence_rarity')}
    config=configs['sequence_uniform'];settings=config['model']
    assert all(c['model']==settings for c in configs.values())
    data=Path(config['data']['loader_kwargs']['sealed_records_by_split']['train'])
    rows=json.loads(data.read_text());assert len(rows)==1173
    token=AutoTokenizer.from_pretrained(settings['llm_name'],local_files_only=True)
    artifacts={}
    for filename in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt'):
        path=cached_file(settings['llm_name'],filename,local_files_only=True,_raise_exceptions_for_missing_entries=False)
        if path:artifacts[str(Path(path).resolve())]=sha(path)
    assert artifacts and token.eos_token_id is not None
    frequencies=Counter(normalize_answer(row['answer']) for row in rows)
    records=[];question_overflow=[]
    for index,row in enumerate(rows):
        prompt=BoneVQAModel._format_llm_prompt(row['question'])
        prompt_ids=token(prompt,add_special_tokens=False,truncation=False)['input_ids']
        if len(prompt_ids)>settings['max_question_tokens']+8:question_overflow.append(index)
        if row['answer_type']!='OPEN':continue
        full=token(' '+row['answer'],add_special_tokens=False,truncation=False)['input_ids']
        limited=token(' '+row['answer'],add_special_tokens=False,truncation=True,
                      max_length=settings['max_answer_tokens'])['input_ids']
        records.append({'TRAIN_index':index,'image_id':row['image_id'],'question':row['question'],
            'answer_token_length':len(full),'training_target_text_tokens_retained':len(limited),
            'training_text_truncated':full!=limited,'target_tokens_including_EOS':len(limited)+1,
            'text_exceeds_generation_budget':len(full)>settings['max_new_tokens'],
            'EOS_plus_text_exceeds_generation_budget':len(full)+1>settings['max_new_tokens'],
            'full_decode_normalized_text_matches_GT':normalize_answer(token.decode(full,skip_special_tokens=True).strip())==normalize_answer(row['answer']),
            'global_TRAIN_answer_frequency_le5':frequencies[normalize_answer(row['answer'])]<=5})
    assert len(records)==534 and sum(r['global_TRAIN_answer_frequency_le5'] for r in records)==72
    report={'passed':True,'scope':'Tokenizer-length arithmetic only on existing TRAIN; not clinical labels or quality evidence',
        'source_script_sha256':sha(Path(__file__).resolve()),'runtime_source_sha256':sha(ROOT/'refactor_v2/src/models/bonevqa.py'),
        'TRAIN_path':str(data),'TRAIN_sha256':sha(data),'tokenizer_artifact_sha256':artifacts,
        'config_sha256':{m:sha(folder/(m+'.json')) for m in configs},
        'max_answer_tokens':settings['max_answer_tokens'],'max_new_tokens':settings['max_new_tokens'],
        'prompt_limit':settings['max_question_tokens']+8,'TRAIN_QA':1173,'TRAIN_OPEN_QA':534,
        'answer_token_length_max':max(r['answer_token_length'] for r in records),
        'training_text_truncated_rows':sum(r['training_text_truncated'] for r in records),
        'text_exceeds_generation_budget_rows':sum(r['text_exceeds_generation_budget'] for r in records),
        'EOS_plus_text_exceeds_generation_budget_rows':sum(r['EOS_plus_text_exceeds_generation_budget'] for r in records),
        'prompt_overflow_indices':question_overflow,
        'normalized_full_decode_mismatch_rows':sum(not r['full_decode_normalized_text_matches_GT'] for r in records),
        'records':records,'model_instantiations':0,'training_updates':0,'reserved_role_inference':False,
        'interpretation':'If truncation counts are zero, configured text length cannot explain these TRAIN failures; this does not prove model capacity or semantic correctness',
        'long_training_authorized':False,'step3_complete':False}
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','tokenizer_artifact_sha256')},indent=2))


if __name__=='__main__':main()
