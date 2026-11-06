"""Summarize gate receipts without promoting them to a training release."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'reports/lapa_pilot_gates_v1'
arms=['carrier','train_cs0','train_cs1','random_cs0','random_cs1']
receipts={a:json.loads((ROOT/a/'receipt.json').read_text()) for a in arms}
assert len({r['initial_state_digest'] for r in receipts.values()})==1
assert all(r['updates']==2 and r['frozen_unchanged'] and r['replay']['exact_logits'] and r['replay']['strict'] for r in receipts.values())
assert all(r['microbatches']==receipts['carrier']['microbatches'] for r in receipts.values())
summary={'all_five_gates_passed':True,'identical_initial_state':True,'matched_microbatches':True,
         'seconds':{a:r['seconds'] for a,r in receipts.items()},
         'update_seconds':{a:[s['seconds'] for s in r['steps']] for a,r in receipts.items()},
         'quality_released':False,
         'remaining':['Save and seal shared PRE-update initialization for quality runner; gate checkpoints are post-update and must not initialize quality runs.',
                      'Materialize quality configs and complete dependency release.',
                      'Implement full-validation outputs, metrics and replay coordinator.',
                      'Audit per-module update attribution beyond the gate LoRA/head assertions.'],
         'receipt_hashes':{a:hashlib.sha256((ROOT/a/'receipt.json').read_bytes()).hexdigest() for a in arms}}
with (ROOT/'gate_summary.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary))
