"""Explicit author LM/current Transformers cache-position compatibility."""
import ast
import copy
import types
from experiments.favp_visual_bridge_v2 import SOURCE, verify_source


def enable_generation_cache_position(model):
    path=SOURCE/'favp/models/modeling_llama.py'
    verify_source(path)
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LlamaForCausalLM')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward'))
    assert 'cache_position' not in [a.arg for a in fn.args.args]
    fn.decorator_list=[]
    fn.args.args.append(ast.arg(arg='cache_position'))
    fn.args.defaults.append(ast.Constant(None))
    count=0
    for node in ast.walk(fn):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name):
            if node.func.value.id=='self' and node.func.attr=='model':
                assert not any(k.arg=='cache_position' for k in node.keywords)
                node.keywords.append(ast.keyword(arg='cache_position',value=ast.Name(id='cache_position',ctx=ast.Load())))
                count+=1
    assert count==1
    namespace=dict(model.forward.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),namespace)
    model.forward=types.MethodType(namespace['forward'],model)
