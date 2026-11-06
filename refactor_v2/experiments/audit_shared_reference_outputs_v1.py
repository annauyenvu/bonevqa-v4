"""Independent saved-logit recount. No model execution or threshold fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]/'reports'
STUDY=ROOT/'lapa_shared_quality_v1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('arm');args=parser.parse_args()
    out=STUDY/args.arm/'independent_metric_audit.json';assert not out.exists()
    vocab=json.loads((ROOT/'step4_localization_v3_1/vocab.json').read_text())
    yes,no=vocab.index('yes'),vocab.index('no');results={}
    for epoch in range(1,5):
        path=STUDY/args.arm/f'epoch{epoch}.json';data=json.loads(path.read_text());rows=data['rows']
        assert len(rows)==1432 and data['updates']==73*epoch
        z=np.array([r['logits'] for r in rows]);assert z.shape==(1432,41) and np.isfinite(z).all()
        predictions=[vocab[i] for i in z.argmax(1)]
        assert predictions==[r['prediction'] for r in rows]
        p=1/(1+np.exp(-(z[:,yes]-z[:,no])))
        assert np.allclose(p,[r['fracture_probability'] for r in rows],rtol=1e-6,atol=1e-7)
        selected=np.array([r['question']=='Is there a fracture in this X-ray?' for r in rows])
        assert selected.sum()==321
        y=np.array([r['answer']=='yes' for r in rows])[selected];scores=p[selected]
        sens=float((scores[y]>=.5).mean());spec=float((scores[~y]<.5).mean())
        calculated={'accuracy':float(np.mean([a==r['answer'] for a,r in zip(predictions,rows)])),
                    'AP':float(average_precision_score(y,scores)),'AUROC':float(roc_auc_score(y,scores)),
                    'sensitivity':sens,'specificity':spec,'BA':(sens+spec)/2,
                    'unsupported':sum(r['answer'] not in vocab for r in rows)}
        for key,value in calculated.items():assert abs(value-data['metrics'][key])<1e-7,(epoch,key)
        results[str(epoch)]={'metrics':calculated,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    with out.open('x') as f:json.dump({'arm':args.arm,'epochs':results,'saved_logits_recount_passed':True,
                                    'not_full_experiment_closure':True},f,indent=2)
    print(json.dumps({'arm':args.arm,'recount_passed':True,'LAST4':results['4']['metrics']}))


if __name__=='__main__':main()
