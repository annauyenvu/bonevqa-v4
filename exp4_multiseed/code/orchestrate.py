from common import *
import time,traceback,fcntl
from engine import csvwrite,normalize_answer,normalize_yes_no

def child(mode,dest,seed=42,regime='auto',variant='vlp',checkpoint=None):
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,str(CODE/'worker.py'),mode,'--dest',str(dest),'--seed',str(seed),'--regime',regime,'--variant',variant]
 if checkpoint:cmd+=['--checkpoint',str(checkpoint)]
 with (dest/'stdout.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONHASHSEED':str(seed)},check=True)

def compare_evaluations(a,b,output):
 x=read(a/'test_predictions.json');y=read(b/'test_predictions.json');assert len(x)==len(y)==2773
 fields=['image_id','question','gt','answer_type','pred','pred_type']
 mismatch=sum(any(p[k]!=q[k] for k in fields) for p,q in zip(x,y));confidence_mismatch=sum(p['confidence']!=q['confidence'] for p,q in zip(x,y))
 closed_mismatch=sum(p['answer_type']=='CLOSED' and p['pred']!=q['pred'] for p,q in zip(x,y))
 correctness_mismatch=sum((normalize_yes_no(p['pred'])==normalize_yes_no(p['gt']))!=(normalize_yes_no(q['pred'])==normalize_yes_no(q['gt'])) if p['answer_type']=='CLOSED' else (normalize_answer(p['pred'])==normalize_answer(p['gt']))!=(normalize_answer(q['pred'])==normalize_answer(q['gt'])) for p,q in zip(x,y))
 report={'qa':2773,'closed_qa':1546,'prediction_mismatch':mismatch,'closed_prediction_mismatch':closed_mismatch,'exact_correctness_mismatch':correctness_mismatch,'confidence_mismatch':confidence_mismatch,'closed_correct_a':read(a/'metrics.json')['n_closed_correct'],'closed_correct_b':read(b/'metrics.json')['n_closed_correct'],'confidence_max_abs_diff':max(abs(p['confidence']-q['confidence']) for p,q in zip(x,y)),'gate_rule':'identical CLOSED predictions and identical per-question exact correctness; OPEN string and confidence discrepancies are audited but do not block the prespecified CLOSED primary endpoint','passed':closed_mismatch==0 and correctness_mismatch==0};write(output,report)
 if not report['passed']:raise RuntimeError('Evaluator repeatability gate failed: '+str(report))
 return report

def main():
 mutex=(ROOT/'pipeline.lock').open('w');fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status('waiting_for_ROI_bank');roi_pid=read(ROOT/'roi_pid.json')['pid']
 while not (ROOT/'roi_bank/COMPLETE.json').exists():
  if (ROOT/'roi_bank/FAILURE.json').exists():raise RuntimeError('ROI preflight failed: '+str(read(ROOT/'roi_bank/FAILURE.json')))
  try:os.kill(roi_pid,0)
  except ProcessLookupError:raise RuntimeError('ROI worker exited without completing bank')
  time.sleep(20)
 bank=read(ROOT/'roi_bank/COMPLETE.json');assert bank['roi_failed']==0
 for regime,r in bank['regimes'].items():
  assert sha(ROOT/'roi_bank'/regime/'index.json')==r['index_sha256'];assert sha(ROOT/'roi_bank'/regime/'audit.json')==r['audit_sha256']
  a=read(ROOT/'roi_bank'/regime/'audit.json');assert a['expected_images']==a['processed_images']==2023 and a['roi_failed']==0
 verify_lock()
 # No training model starts until two independent full evaluator passes agree.
 pre=ROOT/'sanity/evaluator';checkpoint=LEGACY/'checkpoints/abl3_du3/best.pt'
 if not (pre/'GATE.json').exists():
  for repeat in ['a','b']:
   status('evaluator_repeatability',repeat=repeat);child('eval',pre/repeat,checkpoint=checkpoint)
  compare_evaluations(pre/'a',pre/'b',pre/'GATE.json')
 else:assert read(pre/'GATE.json')['passed']
 primary=ROOT/'auto/vlp/seed_42';repeat=ROOT/'sanity/auto_vlp_seed42_repeat'
 for dest in [primary,repeat]:
  if not (dest/'QC.json').exists():status('training_sanity',run=str(dest.relative_to(ROOT)));child('train',dest)
  assert read(dest/'QC.json')['valid'],'Existing technical failure must be reviewed before retry'
 # Reload the same newly selected checkpoint independently as a second same-checkpoint evaluation.
 evalrepeat=ROOT/'sanity/new_checkpoint_repeat'
 if not (evalrepeat/'metrics.json').exists():child('eval',evalrepeat,checkpoint=primary/'checkpoint_best.pt')
 compare_evaluations(primary,evalrepeat,ROOT/'sanity/new_checkpoint_eval_gate.json')
 repeated=compare_evaluations(primary,repeat,ROOT/'sanity/training_prediction_gate.json')
 q1=read(primary/'QC.json');q2=read(repeat/'QC.json');h1=read(primary/'training_history.json');h2=read(repeat/'training_history.json')
 report={'primary_selection':'first run always primary; repeat never selected by score','final_validation_metric_a':h1[-1]['val_overall_acc'],'final_validation_metric_b':h2[-1]['val_overall_acc'],'selected_validation_metric_a':q1['selected_val_overall_acc'],'selected_validation_metric_b':q2['selected_val_overall_acc'],'checkpoint_file_hash_a':sha(primary/'checkpoint_best.pt'),'checkpoint_file_hash_b':sha(repeat/'checkpoint_best.pt'),'checkpoint_tensor_hash_a':(primary/'checkpoint_tensor_hash.txt').read_text().strip(),'checkpoint_tensor_hash_b':(repeat/'checkpoint_tensor_hash.txt').read_text().strip(),**repeated}
 report['passed']=report['passed'] and report['final_validation_metric_a']==report['final_validation_metric_b'] and report['selected_validation_metric_a']==report['selected_validation_metric_b'] and report['checkpoint_tensor_hash_a']==report['checkpoint_tensor_hash_b']
 write(ROOT/'sanity/FULL_MATRIX_GATE.json',report)
 if not report['passed']:raise RuntimeError('Independent-training sanity gate failed; no remaining matrix runs permitted')
 done=0
 for seed in SEEDS:
  for regime in REGIMES:
   for variant in CONFIGS:
    dest=ROOT/regime/variant/f'seed_{seed}'
    if (dest/'QC.json').exists():assert read(dest/'QC.json')['valid'],'Refusing automatic rerun of failed run'
    else:status('primary_matrix',completed=done,total=30,regime=regime,config=variant,seed=seed);child('train',dest,seed,regime,variant)
    done+=1;write(ROOT/'progress.json',{'completed_primary_runs':done,'total_primary_runs':30,'last_completed':str(dest.relative_to(ROOT))})
 status('statistical_analysis',completed_primary_runs=30)
 with (ROOT/'logs/analysis.log').open('a') as log:subprocess.run([sys.executable,str(CODE/'analyze.py')],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONHASHSEED':'42'},check=True)
 status('EXP4','complete',completed_primary_runs=30)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  write(ROOT/'TECHNICAL_FAILURE.json',{'time':datetime.datetime.now().isoformat(),'error':repr(exc),'traceback':traceback.format_exc(),'policy':'No automatic score-based or technical restart; diagnose and record before resuming'});status('EXP4','technical_failure',error=repr(exc));raise
