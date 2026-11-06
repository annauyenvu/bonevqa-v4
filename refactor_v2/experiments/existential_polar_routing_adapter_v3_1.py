"""Experimental question-only routing adapter; never edits frozen sources.

Narrow English existential-presence pattern, not a general choice parser.
Only activate inside an explicitly bound, single-process diagnostic context.
"""
from contextlib import contextmanager
import re
from src.utils import is_closed_question as frozen_closed,is_polar_question


def candidate_closed_question(question):
    normalized=' '.join((question or '').casefold().split())
    existential=re.match(r'^(?:is|are) there any\b',normalized) is not None
    clause_alternative=re.search(r'\bor\s+(?:is|are|was|were|does|do|did|can|could|would|will|should)\b',normalized) is not None
    if existential and not clause_alternative and is_polar_question(normalized):
        return True
    return frozen_closed(question)


@contextmanager
def experimental_routing():
    import src.models.bonevqa as model_module
    assert model_module.is_closed_question is frozen_closed, 'No nesting or competing routing patch'
    model_module.is_closed_question=candidate_closed_question
    try:
        yield
    finally:
        model_module.is_closed_question=frozen_closed


def test_contract():
    # Only nonclinical synthetic grammar fixtures. They are unit-test expected
    # routes, NOT generated clinical labels or additions to a dataset.
    cases=[('Is there any pen or pencil in the box?',True),
           ('Are there any pens or pencils in the box?',True),
           ('  IS there any pen OR pencil in the box?  ',True),
           ('Is the object red or blue?',False),
           ('Is there any pen or is the box empty?',False),
           ('Which object is red?',False),('Is the box empty?',True),
           ('Are the objects red or not?',True),('',False),(None,False)]
    for question,expected in cases:assert candidate_closed_question(question)==expected,(question,expected)
    import src.models.bonevqa as model_module
    original=model_module.is_closed_question
    with experimental_routing():
        assert model_module.is_closed_question is candidate_closed_question
        nested_rejected=False
        try:
            with experimental_routing():pass
        except AssertionError:nested_rejected=True
        assert nested_rejected
    assert model_module.is_closed_question is original
    try:
        with experimental_routing():raise RuntimeError('unit-test sentinel')
    except RuntimeError:pass
    assert model_module.is_closed_question is original
    return {'grammar_fixtures':len(cases),'nested_adapter_rejected':True,
            'restored_on_normal_and_exception_exit':True,'model_instantiations':0}
