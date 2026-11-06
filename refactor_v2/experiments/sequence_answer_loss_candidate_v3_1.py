"""Pure loss candidate, not installed in frozen model/training runtime."""
import torch
import torch.nn.functional as F


def sequence_losses(logits,labels,weights=None):
    assert logits.ndim==3 and labels.shape==logits.shape[:2]
    shifted_logits=logits[:,:-1,:].float().contiguous()
    shifted_labels=labels[:,1:].contiguous()
    losses=F.cross_entropy(shifted_logits.reshape(-1,shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),ignore_index=-100,reduction='none').reshape_as(shifted_labels)
    valid=shifted_labels.ne(-100);counts=valid.sum(dim=1)
    assert bool((counts>0).all()), 'Every OPEN sequence must contain supervised answer/EOS tokens'
    per_sequence=(losses*valid).sum(dim=1)/counts
    if weights is None:weights=torch.ones_like(per_sequence)
    weights=torch.as_tensor(weights,device=logits.device,dtype=per_sequence.dtype)
    assert weights.shape==per_sequence.shape and bool(torch.isfinite(weights).all()) and bool((weights>0).all())
    return {'token_mean':(losses*valid).sum()/counts.sum(),
            'sequence_uniform':per_sequence.mean(),
            'sequence_rarity':(per_sequence*weights).mean()}


def numeric_unit_gate():
    # Integer-token fixtures only, not clinical labels or dataset examples.
    generator=torch.Generator().manual_seed(20261020)
    logits=torch.randn(2,6,7,generator=generator,requires_grad=True)
    labels=torch.tensor([[-100,-100,2,3,4,1],[-100,-100,-100,-100,2,1]])
    output=sequence_losses(logits,labels,[.5,1.5])
    reference=F.cross_entropy(logits[:,:-1,:].contiguous().reshape(-1,7),labels[:,1:].contiguous().reshape(-1),ignore_index=-100)
    assert torch.allclose(reference,output['token_mean'],atol=1e-7,rtol=1e-7)
    a=torch.autograd.grad(reference,logits,retain_graph=True)[0]
    b=torch.autograd.grad(output['token_mean'],logits,retain_graph=True)[0]
    assert torch.allclose(a,b,atol=1e-7,rtol=1e-7)
    assert sequence_losses(logits,labels)['sequence_rarity'].equal(output['sequence_uniform'])
    padded_logits=torch.cat([logits,torch.zeros(2,2,7)],dim=1)
    padded_labels=torch.cat([labels,torch.full((2,2),-100)],dim=1)
    padded=sequence_losses(padded_logits,padded_labels,[.5,1.5])
    assert all(torch.allclose(output[k],padded[k],atol=1e-7,rtol=1e-7) for k in output)
    invalid=False
    try:sequence_losses(logits,torch.full_like(labels,-100))
    except AssertionError:invalid=True
    assert invalid
    assert all(torch.isfinite(v) for v in output.values())
    return {'passed':True,'integer_token_unit_fixtures_only':True,
        'token_mean_matches_native_CE_values_and_gradients':True,
        'uniform_weights_recover_sequence_uniform':True,'ignored_padding_invariant':True,
        'unsupervised_sequence_rejected':True,'actual_Qwen_model_gradient_gate_pending':True}
