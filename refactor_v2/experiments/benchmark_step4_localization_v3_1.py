"""Fresh-process initialization audit and bounded representative throughput."""
import gc,time,json
import numpy as np
import torch
from src.models.bonevqa import BoneVQAModel
from src.data import build_dataset,collate_fn
from src.data.fracatlas_vqa import FracAtlasVQADataset
from src.train import build_split,seed_training_epoch
from experiments.prepare_step4_localization_v3_1 import OUT,read,dump
from experiments.gate_step4_localization_v3_1 import hashes
from experiments.vl_bounded_proposal_contract_v3_1 import sha

def main():
    torch.set_num_threads(8); gate=read(OUT/'gate_report.json'); assert gate['passed']
    assert not (OUT/'benchmark.json').exists()
    schedules=read(OUT/'schedules.json'); result={}
    for arm in ('B','V_real','V_placebo'):
        cfg=read(OUT/(arm+'.json')); path=OUT/(arm+'_PREtask.pt')
        assert sha(path)==gate['arms'][arm]['initialization_sha256']
        model=BoneVQAModel.load(str(path),device='cpu')
        assert hashes(model.trainable_state_dict())==gate['arms'][arm]['initial_tensor_hashes']
        datasets={r:build_split(build_dataset,cfg['data']['dataset'],r,None,True,cfg['data']['loader_kwargs']) for r in ('train','model_val')}
        boxes=read(OUT/('placebo.json' if arm=='V_placebo' else 'boxes.json'))
        def batch(role,indices):
            b=collate_fn([datasets[role][i] for i in indices])
            if arm!='B':
                b['boxes']=[boxes[i] or None for i in b['image_ids']]
                b['masks']=[FracAtlasVQADataset._boxes_to_masks(boxes[i],im.size) for i,im in zip(b['image_ids'],b['images'])]
            return b
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
        model.train(); seed_training_epoch(42,0); optimizer.zero_grad(set_to_none=True)
        timings=[]
        for step,index in enumerate(np.linspace(0,len(schedules[0])-1,16,dtype=int)):
            t=time.perf_counter(); b=batch('train',schedules[0][index]); o=model(b,stage=3)
            assert torch.isfinite(o['loss']); (o['loss']/4).backward()
            if (step+1)%4==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            timings.append(time.perf_counter()-t)
        model.eval(); eval_times=[]
        for index in np.linspace(0,len(datasets['model_val'])-4,12,dtype=int):
            t=time.perf_counter(); b=batch('model_val',range(index,index+4))
            public={k:b[k] for k in ('images','questions','boxes','masks') if k in b and (arm!='B' or k in ('images','questions'))}
            with torch.no_grad(): model.predict_batch(public)
            eval_times.append(time.perf_counter()-t)
        train_est=sum(map(len,schedules))*float(np.mean(timings))
        eval_est=5*((len(datasets['model_val'])+3)//4)*float(np.mean(eval_times))
        result[arm]={'fresh_process_initialization_exact':True,'disposable_updates':4,'train_microbatch_seconds':timings,'eval_batch_seconds':eval_times,'estimated_train_seconds':train_est,'estimated_five_evaluations_seconds':eval_est,'estimated_compute_seconds':train_est+eval_est,'scope':'16 distributed TRAIN microbatches and 12 distributed model-validation batches, disposable weights; estimate excludes checkpoint I/O/startup and full replays'}
        assert sha(path)==gate['arms'][arm]['initialization_sha256']
        print(json.dumps({'arm':arm,'estimated_compute_seconds':train_est+eval_est}),flush=True)
        del model,optimizer,o; gc.collect()
    dump(OUT/'benchmark.json',{'passed':True,'arms':result,'estimated_compute_seconds_total':sum(r['estimated_compute_seconds'] for r in result.values()),'training_released':False})
if __name__=='__main__': main()
