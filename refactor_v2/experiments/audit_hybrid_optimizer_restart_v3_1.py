"""Checkpoint/optimizer/hash audit without neural model instantiation."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/hybrid_optimizer_restart_gate_v3_1'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def statehash(value):
    import torch
    digest=hashlib.sha256()
    def walk(x):
        if torch.is_tensor(x):
            digest.update(str((str(x.dtype),list(x.shape))).encode());digest.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=repr):digest.update(repr(key).encode());walk(x[key])
        elif isinstance(x,(tuple,list)):
            for entry in x:walk(entry)
        else:digest.update(repr(x).encode())
    walk(value);return digest.hexdigest()


if __name__=='__main__':
    destination=OUT/'independent_artifact_audit.json';assert not destination.exists()
    p=read(OUT/'protocol.json');r=read(OUT/'report.json')
    for path,digest in p['dependencies_sha256'].items():assert sha(path)==digest,path
    assert r['passed'] and r['actual_optimizer_updates']==3
    assert r['protocol_sha256']==sha(OUT/'protocol.json')
    assert r['reference_sha256']==sha(OUT/'continuous_reference.json')
    assert r['restart_sha256']==sha(OUT/'fresh_process_restart.json')
    assert r['disposable_export_sha256']==sha(OUT/'DISPOSABLE_after_update1.pt')
    assert r['disposable_optimizer_sha256']==sha(OUT/'DISPOSABLE_optimizer1.pt')
    assert sha(p['initial_checkpoint'])==p['initial_checkpoint_sha256']
    restart=read(OUT/'fresh_process_restart.json')
    assert restart['passed'] and restart['all263final_parameter_hashes_exact']
    assert restart['optimizer_scheduler_states_exact'] and restart['second_update_loss_norm_LR_exact']
    assert not r['quality_training_released'] and not r['fullfold_or_mid_epoch_resume_certified']
    import torch
    torch.set_num_threads(8)
    payload=torch.load(OUT/'DISPOSABLE_after_update1.pt',map_location='cpu',weights_only=False)
    original=torch.load(p['initial_checkpoint'],map_location='cpu',weights_only=False)
    optim=torch.load(OUT/'DISPOSABLE_optimizer1.pt',map_location='cpu',weights_only=False)
    ref=read(OUT/'continuous_reference.json')
    hashes=lambda state:{k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in state.items()}
    assert len(payload['state'])==263 and hashes(payload['state'])==ref['after_first_parameter_hashes']
    assert payload['config']==original['config'] and payload['answer_vocab']==original['answer_vocab']
    assert all(torch.isfinite(v).all() for v in payload['state'].values())
    assert payload['extra']['disposable_technical_gate'] and payload['extra']['quality_initialization_forbidden']
    assert payload['extra']['global_step']==1
    assert statehash(optim['optimizer'])==ref['after_first_optimizer_hash']
    assert statehash(optim['scheduler'])==ref['after_first_scheduler_hash']
    original_hashes=hashes(original['state']);final=ref['final_parameter_hashes']
    assert set(final)==set(original_hashes)
    groups={'visual_LoRA':lambda n:n.startswith('visual.') and 'lora_' in n,
            'QFormer':lambda n:n.startswith('qformer.'),'fracture_head':lambda n:n.startswith('fracture_head.'),
            'LLM_LoRA':lambda n:n.startswith('llm.') and 'lora_' in n,
            'closed_head':lambda n:n.startswith('closed_head.')}
    updates={group:{'tensors':sum(match(n) for n in final),
        'changed_tensors':sum(match(n) and final[n]!=original_hashes[n] for n in final)} for group,match in groups.items()}
    assert all(value['changed_tensors']>0 for value in updates.values())
    report={'passed':True,'report_sha256':sha(OUT/'report.json'),'protocol_sha256':sha(OUT/'protocol.json'),
        'auditor_sha256':sha(__file__),'physical_disposable_export_optimizer_hashes_checked':True,
        'first_export_all263tensor_hashes_config_vocabulary_optimizer_scheduler_recomputed':True,
        'fresh_process_second_update_263parameter_optimizer_scheduler_exact_bound':True,
        'original_quality_checkpoint_unchanged':True,'module_disposable_update_counts':updates,
        'neural_models':0,'optimizer_updates_by_auditor':0,'actual_optimizer_updates_in_gate':3,
        'all_updated_weights_disposable_quality_initialization_forbidden':True,
        'quality_training_released':False,'fullfold_or_mid_epoch_resume_certified':False}
    dump=json.dumps(report,indent=2)+'\n';destination.write_text(dump);print(dump)
