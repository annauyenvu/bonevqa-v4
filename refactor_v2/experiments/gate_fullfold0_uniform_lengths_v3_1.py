"""Predetermined heterogeneous TRAIN lengths, four disposable optimizer steps."""
import json
from contextlib import contextmanager
from pathlib import Path
import torch
from transformers import AutoTokenizer
from experiments.audit_fresh_exact_upstream_fracatlas_v3_1 import ROOT, sha
from experiments.run_vl_bounded_pilot_v3_1 import check_environment
from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss as original_adapter
from experiments.sequence_answer_loss_candidate_v3_1 import sequence_losses


@contextmanager
def observed_adapter(model,batch,stage,mode,table):
    if not model.training or stage!=3:
        with original_adapter(model,batch,stage,mode,table) as trace:
            yield trace
        return
    original=model.llm.forward;captured={}
    def observe(*args,**kwargs):
        result=original(*args,**kwargs)
        labels=kwargs['labels']
        lengths=labels[:,1:].ne(-100).sum(dim=1).tolist()
        assert len(lengths)==2 and min(lengths)<max(lengths)
        captured.update(actual_target_lengths=lengths,
            diagnostic_uniform=sequence_losses(result.logits,labels)['sequence_uniform'])
        return result
    model.llm.forward=observe
    try:
        with original_adapter(model,batch,stage,mode,table) as trace:
            yield trace
            assert trace['calls']==1
            trace.update(captured)
    finally:
        model.llm.forward=original


if __name__=='__main__':
    check_environment()
    prior=ROOT/'refactor_v2/reports/fullfold0_uniform_timing_gate_v3_3/protocol.json'
    assert sha(prior)=='a00f26a1017768d6a8c41ecd4d314a13c1f8a75ea35dbc13b4e5f13a6696a462'
    for name,h in json.loads(prior.read_text())['dependencies_sha256'].items():assert sha(Path(name))==h
    base=ROOT/'.evaluation_runtime/data/fracatlas_component_cv_v3/fold_0'
    rows=json.loads((base/'vqa_train.json').read_text());groups=json.loads((base/'detector_split_v3.json').read_text())['component_by_image']
    tokenreport=ROOT/'refactor_v2/reports/TRAIN_token_budget_v3_1.json'
    tokenizer_artifacts=json.loads(tokenreport.read_text())['tokenizer_artifact_sha256']
    for name,h in tokenizer_artifacts.items():assert sha(Path(name))==h
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct',local_files_only=True)
    lengths={i:len(tok(' '+r['answer'],add_special_tokens=False)['input_ids']) for i,r in enumerate(rows) if r['answer_type']=='OPEN'}
    selected=[];used=set()
    for answer in ('yes','no'):
        i=next(i for i,r in enumerate(rows) if r['question']=='Is there a fracture in this X-ray?' and r['answer']==answer and groups[r['image_id']] not in used)
        selected.append(i);used.add(groups[rows[i]['image_id']])
    short=min((i for i in lengths if groups[rows[i]['image_id']] not in used),key=lambda i:(lengths[i],i))
    selected.append(short);used.add(groups[rows[short]['image_id']])
    long=min((i for i in lengths if groups[rows[i]['image_id']] not in used),key=lambda i:(-lengths[i],i))
    selected.append(long);used.add(groups[rows[long]['image_id']])
    assert len(used)==4 and 0<lengths[short]<lengths[long]<=32
    selection={'policy':'First canonical yes/no on unused components; shortest OPEN then longest OPEN on unused components, token length then original index; no prediction-based selection',
        'OPEN_text_token_lengths':[lengths[short],lengths[long]],'expected_supervised_lengths_including_EOS':[lengths[short]+1,lengths[long]+1],
        'tokenizer_files_sha256':tokenizer_artifacts}
    source=ROOT/'refactor_v2/experiments/gate_fullfold0_uniform_timing_v3_1.py';text=source.read_text()
    start=text.index('    chosen=[];used=set()\n');end=text.index('    batch=collate_fn',start)
    text=text[:start]+"    chosen=list(PRESELECTED);used={lookup[rows[i]['image_id']] for i in chosen};assert len(used)==4\n"+text[end:]
    replacements={
        "OUT=ROOT/'refactor_v2/reports/fullfold0_uniform_timing_gate_v3_1'":"OUT=ROOT/'refactor_v2/reports/fullfold0_uniform_lengths_v3_1'",
        "for p in (train,vocabpath,base/'detector_split_v3.json'):":"for p in (train,vocabpath):",
        "root=str(ROOT/'data'),":"root=str(ROOT/'.evaluation_runtime/data/fracatlas/FracAtlas'),",
        "[Path(__file__).resolve(),cfgpath,train,vocabpath,sealpath,":"[Path(__file__).resolve(),Path(SOURCE_CORE_PATH),Path(PRIOR_PROTOCOL),Path(TOKEN_REPORT),*map(Path,TOKEN_FILES),cfgpath,train,vocabpath,sealpath,base/'detector_split_v3.json',",
        "'selection':'first canonical yes/no then first two OPEN on unused components, no prediction selection',":"'selection':SELECTION_METADATA,",
        "                grads=torch.autograd.grad(out['loss_answer']":"                native_grads=torch.autograd.grad(trace['native_loss'],[params[n] for n in names],retain_graph=True)\n                token_grads=torch.autograd.grad(trace['token_mean'],[params[n] for n in names],retain_graph=True)\n                uniform_grads=torch.autograd.grad(trace['diagnostic_uniform'],[params[n] for n in names],retain_graph=True)\n                native_errors={n:float((a-b).abs().max()) for n,a,b in zip(names,native_grads,token_grads)}\n                for a,b in zip(native_grads,token_grads):assert torch.isfinite(a).all() and torch.isfinite(b).all() and torch.allclose(a,b,atol=1e-6,rtol=1e-4)\n                uniform_differences={n:float((a-b).abs().max()) for n,a,b in zip(names,native_grads,uniform_grads)}\n                assert all(torch.isfinite(g).all() for g in uniform_grads)\n                if step==0:assert max(uniform_differences.values())>1e-8 and abs(float(trace['native_loss'].detach()-trace['diagnostic_uniform'].detach()))>1e-6\n                assert trace['actual_target_lengths']==SELECTION_METADATA['expected_supervised_lengths_including_EOS']\n                grads=torch.autograd.grad(out['loss_answer']",
        "steps.append({'loss':loss,'answer_gradient_max':maxima,'seconds':time.perf_counter()-begin})":"steps.append({'loss':loss,'native_LM':float(trace['native_loss'].detach()),'uniform_LM':float(trace['diagnostic_uniform'].detach()),'actual_target_lengths':trace['actual_target_lengths'],'native_token_gradient_max_errors':native_errors,'native_uniform_gradient_max_differences':uniform_differences,'answer_gradient_max':maxima,'seconds':time.perf_counter()-begin})",
        "del out,trace,grads":"del out,trace,grads,native_grads,token_grads,uniform_grads"}
    for old,new in replacements.items():
        assert text.count(old)==1,old
        text=text.replace(old,new)
    namespace={'__name__':'fullfold0_length_gate','__file__':str(Path(__file__).resolve()),
        'SOURCE_CORE_PATH':str(source),'PRIOR_PROTOCOL':str(prior),'TOKEN_REPORT':str(tokenreport),
        'TOKEN_FILES':list(tokenizer_artifacts),'PRESELECTED':selected,'SELECTION_METADATA':selection}
    exec(compile(text,str(source),'exec'),namespace)
    namespace['private_training_loss']=observed_adapter
    namespace['main']()
