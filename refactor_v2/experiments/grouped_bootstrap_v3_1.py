"""Within-outer-fold known-component bootstrap; paired outcomes use identical indices."""
import numpy as np

RESAMPLES=2000
SEED=20261021

def groups(rows):
    if not rows:raise ValueError('Empty evaluation records')
    result={};owners={}
    for i,row in enumerate(rows):
        fold=row['fold'];component=row['component']
        if not isinstance(fold,str) or not fold or not isinstance(component,str) or not component:
            raise ValueError('Explicit fold and known-component identifiers required')
        if component in owners and owners[component]!=fold:
            raise ValueError('Known component crosses outer folds')
        owners[component]=fold
        result.setdefault(fold,{}).setdefault(component,[]).append(i)
    return {fold:[components[c] for c in sorted(components)] for fold,components in sorted(result.items())}

def index_draws(rows):
    partition=groups(rows);rng=np.random.default_rng(SEED)
    for _ in range(RESAMPLES):
        indices=[]
        for fold,components in partition.items():
            selected=rng.integers(0,len(components),size=len(components))
            for k in selected:indices.extend(components[int(k)])
        yield indices

def summarize(draw_values):
    # Keep undefined draws visible rather than replacing them by zero.
    finite=[float(v) for v in draw_values if v is not None and np.isfinite(v)]
    return {'valid_draws':len(finite),'undefined_draws':len(draw_values)-len(finite),
            'percentile_2_5_97_5':np.percentile(finite,[2.5,97.5]).tolist() if finite else None}

def bootstrap(rows,score):
    values={}
    for indices in index_draws(rows):
        measured=score([rows[i] for i in indices])
        if values and set(measured)!=set(values):raise ValueError('Metric set changed across draws')
        for key,value in measured.items():values.setdefault(key,[]).append(value)
    return {'resamples':RESAMPLES,'seed':SEED,'intervals':{k:summarize(v) for k,v in values.items()},
            'draw_values':values,'scope':'Conditional on frozen models and training seed; known components are not verified patient identities'}
