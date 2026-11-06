"""Bounded classifier pilot worker. Requires an explicit verified release."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from experiments.lapa_pilot_runner_v1 import PilotRunner, ARMS
from experiments.lapa_reference_endpoint_v1 import reference_loss
from experiments.lapa_integrated_gate_v1 import REPORTS
from experiments.prepare_lapa_pilot_v1 import sha
from src.data import build_dataset, collate_fn
from src.train import build_split
from src.utils import normalize_answer

OUT=REPORTS/'lapa_shared_quality_v1'
PREP=REPORTS/'lapa_pilot_preparation_v1'
CANONICAL='Is there a fracture in this X-ray?'


def metrics(rows):
    correct=sum(r['prediction']==r['answer'] for r in rows)
    canonical=[r for r in rows if r['question']==CANONICAL]
    assert len({r['image_id'] for r in canonical})==len(canonical)
    y=[int(r['answer']=='yes') for r in canonical]; p=[r['fracture_probability'] for r in canonical]
    tp=sum(a==1 and b>=.5 for a,b in zip(y,p)); tn=sum(a==0 and b<.5 for a,b in zip(y,p))
    pos=sum(y); neg=len(y)-pos
    families={q: {'count':sum(r['question']==q for r in rows),
                  'accuracy':sum(r['prediction']==r['answer'] for r in rows if r['question']==q)/sum(r['question']==q for r in rows)}
              for q in sorted({r['question'] for r in rows})}
    return {'records':len(rows),'accuracy':correct/len(rows),'question_strata':families,'unsupported':sum(not r['supported'] for r in rows),
            'AP':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p)),
            'sensitivity':tp/pos,'specificity':tn/neg,'BA':.5*(tp/pos+tn/neg),
            'fracture_records':len(y),'other_class_argmax':sum(r['prediction'] not in ('yes','no') for r in canonical)}


def evaluate(model,dataset):
    model.eval(); rows=[]
    yes,no=model.vocab.index('yes'),model.vocab.index('no')
    with torch.no_grad():
        for start in range(0,len(dataset),4):
            records=[dataset[i] for i in range(start,min(start+4,len(dataset)))]; batch=collate_fn(records)
            logits,_=model(batch['images'],batch['questions'])
            probs=(logits[:,yes]-logits[:,no]).sigmoid()
            for r,z,p in zip(records,logits,probs):
                answer=normalize_answer(r['answer'])
                rows.append({'image_id':r['image_id'],'question':r['question'],'answer':answer,
                             'prediction':model.vocab[int(z.argmax())],'supported':answer in model.vocab,
                             'fracture_probability':float(p),'logits':z.tolist()})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=list(ARMS));parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    release=json.loads((OUT/'execution_release.json').read_text())
    assert release['quality_released'] is True
    for p,h in release['dependencies'].items():assert sha(Path(p))==h,p
    folder=OUT/args.arm
    if not args.replay:folder.mkdir(exist_ok=False)
    torch.set_num_threads(8);torch.manual_seed(42)
    config=json.loads((PREP/'configs.json').read_text())[args.arm]
    model=PilotRunner(args.arm)
    saved=torch.load(folder/'LAST.pt' if args.replay else config['init'],map_location='cpu')
    model.load_state_dict(saved['state'],strict=True);del saved
    cfg=json.loads((REPORTS/'step4_localization_v3_1/B.json').read_text())
    val=build_split(build_dataset,cfg['data']['dataset'],'model_val',None,True,cfg['data']['loader_kwargs'])
    assert len(val)==1432
    if args.replay:
        rows=evaluate(model,val); expected=json.loads((folder/'epoch4.json').read_text())['rows']
        assert rows==expected
        with (folder/'replay.json').open('x') as f:json.dump({'records':len(rows),'exact':True},f)
        return
    train=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
    schedules=json.loads(Path(config['schedule']).read_text()); opt=model.optimizer(); updates=0
    length=max(len(model.text.tokenizer.encode(a,add_special_tokens=True)) for a in model.vocab)
    for epoch,schedule in enumerate(schedules,1):
        torch.manual_seed(42+epoch);model.train();tick=time.perf_counter(); loss_sum=0.; micro=0
        for offset in range(0,len(schedule),4):
            window=schedule[offset:offset+4];opt.zero_grad()
            for indices in window:
                records=[train[i] for i in indices];batch=collate_fn(records);answers=[normalize_answer(r['answer']) for r in records]
                labels=F.one_hot(torch.tensor([model.vocab.index(a) for a in answers]),41).float()
                ids=model.text.tokenizer(answers,padding='max_length',max_length=length,truncation=False,return_tensors='pt')['input_ids']
                with torch.no_grad():words=model.text.model.get_input_embeddings()(ids)
                logits,generated=model(batch['images'],batch['questions'])
                loss=reference_loss(logits,labels,generated,words,model.eta)['total']
                assert torch.isfinite(loss);(loss/len(window)).backward();loss_sum+=float(loss.detach());micro+=1
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            assert torch.isfinite(norm);opt.step();updates+=1
        train_seconds=time.perf_counter()-tick;tick=time.perf_counter();rows=evaluate(model,val)
        result={'epoch':epoch,'updates':updates,'mean_microbatch_loss':loss_sum/micro,'train_seconds':train_seconds,
                'eval_seconds':time.perf_counter()-tick,'metrics':metrics(rows),'rows':rows}
        with (folder/f'epoch{epoch}.json').open('x') as f:json.dump(result,f)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    assert updates==292
    torch.save({'state':model.state_dict(),'updates':updates},folder/'LAST.pt')
    with (folder/'completed.json').open('x') as f:json.dump({'updates':updates,'epochs':4},f)


if __name__=='__main__':main()
