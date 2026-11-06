"""Allowlisted loader for verified SAMMed2D archive with inert optimizer stub.

Never imports/constructs the serialized optimizer. Unknown globals are rejected.
This handles a training archive that torch2.2 weights_only rejects wholesale.
"""
import collections
import io
import pickle
import types
from pathlib import Path
import torch
from experiments.prepare_lapa_pilot_v1 import sha

EXPECTED='921d8c130cc4bcc4b7220c064e60f207f15ad59f17ab36de340e41c67a490d18'


class InertOptimizer:
    def __init__(self,*args,**kwargs):pass
    def __setstate__(self,state):self.saved_state=state


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('collections','OrderedDict'):collections.OrderedDict,
                 ('collections','defaultdict'):collections.defaultdict,
                 ('__builtin__','dict'):dict,('builtins','dict'):dict,
                 ('torch','FloatStorage'):torch.FloatStorage,
                 ('torch._utils','_rebuild_parameter'):torch._utils._rebuild_parameter,
                 ('torch._utils','_rebuild_tensor_v2'):torch._utils._rebuild_tensor_v2,
                 ('torch.optim.adam','Adam'):InertOptimizer}
        if (module,name) not in allowed:raise pickle.UnpicklingError(f'Rejected global {module}.{name}')
        return allowed[module,name]


def load_archive(path,map_location='meta'):
    path=Path(path);assert sha(path)==EXPECTED
    restricted=types.ModuleType('verified_sam_restricted_pickle')
    restricted.Unpickler=RestrictedUnpickler
    restricted.load=lambda f,**kwargs:RestrictedUnpickler(f,**kwargs).load()
    restricted.loads=lambda data,**kwargs:RestrictedUnpickler(io.BytesIO(data),**kwargs).load()
    # Explicitly supply the restrictive Unpickler: default unrestricted pickle is
    # never used. No safe_globals/optimizer allowlisting executes external code.
    return torch.load(path,map_location=map_location,pickle_module=restricted,weights_only=False)
