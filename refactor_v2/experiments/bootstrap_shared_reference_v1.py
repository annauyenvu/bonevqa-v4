"""Post-run paired component uncertainty for prospectively fixed contrasts."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

REPORTS=Path(__file__).resolve().parents[1]/'reports'
OUT=REPORTS/'lapa_shared_quality_v1'
ARMS=['carrier','train_cs0','train_cs1','random_cs0','random_cs1']
PAIRS=[('train_cs0','carrier'),('train_cs1','carrier'),('train_cs0','random_cs0'),
       ('train_cs1','random_cs1'),('train_cs1','train_cs0'),('random_cs1','random_cs0')]


def main():
    assert (OUT/'execution_completed.json').exists()
    assert not (OUT/'paired_component_bootstrap.json').exists()
    sources={}; probabilities={};accuracies={};ids=None;rows_all=None
    for arm in ARMS:
        assert json.loads((OUT/arm/'replay.json').read_text())['exact']
        path=OUT/arm/'epoch4.json';d=json.loads(path.read_text());assert d['updates']==292
        rows=d['rows'];canonical=[r for r in rows if r['question']=='Is there a fracture in this X-ray?']
        current=[r['image_id'] for r in canonical]
        if ids is None:
            ids=current;y=np.array([r['answer']=='yes' for r in canonical]);rows_all=rows
        assert current==ids and [r['answer']=='yes' for r in canonical]==y.tolist()
        assert [(r['image_id'],r['question'],r['answer']) for r in rows]==[(r['image_id'],r['question'],r['answer']) for r in rows_all]
        probabilities[arm]=np.array([r['fracture_probability'] for r in canonical])
        accuracies[arm]=np.array([r['prediction']==r['answer'] for r in rows])
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    lookup=json.loads((REPORTS/'step4_localization_v3_1/component_lookup.json').read_text())
    groups=sorted({lookup[i] for i in ids})
    image_indices=[np.array([j for j,i in enumerate(ids) if lookup[i]==g],dtype=int) for g in groups]
    qa_indices=[np.array([j for j,r in enumerate(rows_all) if lookup[r['image_id']]==g],dtype=int) for g in groups]
    draws={f'{a}_minus_{b}':{'AP':[],'accuracy':[]} for a,b in PAIRS}
    rng=np.random.default_rng(20261031)
    for _ in range(2000):
        selected=rng.integers(0,len(groups),len(groups))
        ii=np.concatenate([image_indices[i] for i in selected]); qi=np.concatenate([qa_indices[i] for i in selected])
        if len(set(y[ii]))<2:continue
        aps={a:float(average_precision_score(y[ii],probabilities[a][ii])) for a in ARMS}
        accs={a:float(accuracies[a][qi].mean()) for a in ARMS}
        for a,b in PAIRS:
            entry=draws[f'{a}_minus_{b}'];entry['AP'].append(aps[a]-aps[b]);entry['accuracy'].append(accs[a]-accs[b])
    points={a:{'AP':float(average_precision_score(y,probabilities[a])),'accuracy':float(accuracies[a].mean())} for a in ARMS}
    comparison={f'{a}_minus_{b}':{metric:{'difference':points[a][metric]-points[b][metric],
        'CI95':np.quantile(draws[f'{a}_minus_{b}'][metric],[.025,.975]).tolist(),
        'valid_draws':len(draws[f'{a}_minus_{b}'][metric])} for metric in ('AP','accuracy')} for a,b in PAIRS}
    report={'seed':20261031,'requested_draws':2000,'components':len(groups),'points':points,'comparisons':comparison,
            'scope':'Exploratory unadjusted intervals conditional on single-seed trained models. Components are not verified patients; development-informed validation. Closed-vocabulary classifier, not historical generative endpoint.',
            'sources':sources,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (OUT/'paired_component_bootstrap.json').open('x') as f:json.dump(report,f,indent=2)


if __name__=='__main__':main()
