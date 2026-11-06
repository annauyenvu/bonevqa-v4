"""Scoped supervised-TRAIN-only LM loss replacement; no public input change."""
from contextlib import contextmanager
from src.models.bonevqa import normalize_batch
from src.utils import normalize_answer
from experiments.sequence_answer_loss_candidate_v3_1 import sequence_losses


@contextmanager
def private_training_loss(model,batch,stage,mode,table):
    assert mode in ('token_mean','sequence_uniform','sequence_rarity')
    trace={'applied':False,'calls':0}
    # Do not even inspect target/metadata fields on eval/inference paths.
    if not model.training or stage!=3:
        yield trace
        return
    normalized=normalize_batch(batch)
    indices=[i for i,t in enumerate(normalized['answer_types']) if str(t).upper()!='CLOSED']
    if not indices:
        yield trace
        return
    weights=[table[(normalized['questions'][i],normalize_answer(normalized['answers'][i]))] for i in indices]
    assert not getattr(model.llm,'_bonevqa_sequence_loss_active',False),'No nested loss adapters'
    original=model.llm.forward
    model.llm._bonevqa_sequence_loss_active=True
    def forward(*args,**kwargs):
        assert kwargs.get('labels') is not None,'Private TRAIN adapter only accepts supervised answer-token calls'
        output=original(*args,**kwargs)
        losses=sequence_losses(output.logits,kwargs['labels'],weights)
        trace.update(applied=True,calls=trace['calls']+1,native_loss=output.loss,
            token_mean=losses['token_mean'],selected_loss=losses[mode],weights=list(weights))
        output.loss=losses[mode]
        output['loss']=losses[mode]
        return output
    model.llm.forward=forward
    try:
        yield trace
    finally:
        model.llm.forward=original
        delattr(model.llm,'_bonevqa_sequence_loss_active')
