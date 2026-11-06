"""Read-only independent execution artifact checks; no model inference or quality claim."""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path('/Users/uyen/Downloads/BONEVQA')
REPORTS = ROOT / 'refactor_v2/reports'

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main(fold):
    assert fold in ('1', '2', '3', '4')
    run = REPORTS / 'grouped_native_training_v3_1'
    output = REPORTS / f'grouped_fold{fold}_execution_artifact_audit_v3_1.json'
    assert not output.exists(), 'Preserve prior audit; do not overwrite'
    assert fold in read(run / 'progress.json')['completed']
    release = read(run / 'release.json')
    binding = release['bindings'][fold]
    config = read(binding['config'])
    assert sha(binding['config']) == binding['config_sha256']
    folder = Path(binding['checkpoint_dir'])
    assert read(folder / 'config.json') == config
    schedule_path = REPORTS / f'grouped_native_preparation_v3_2/fold_{fold}_schedule.json'
    assert sha(schedule_path) == binding['schedule_sha256']
    schedule = read(schedule_path)
    train = read(config['data']['loader_kwargs']['sealed_records_by_split']['train'])
    modelval = read(config['data']['loader_kwargs']['sealed_records_by_split']['model_val'])
    for epoch in schedule:
        indices = [i for batch in epoch['ordered_indices'] for i in batch]
        assert sorted(indices) == list(range(len(train)))
        assert epoch['updates'] == math.ceil(len(epoch['ordered_indices']) / 4)
    expected = {
        'TRAIN_calls': sum(len(e['ordered_indices']) for e in schedule),
        'TRAIN_rows': len(schedule) * len(train),
        'native_eval_rows': len(schedule) * len(modelval),
    }
    completion = read(run / f'fold_{fold}_completion.json')
    assert completion['passed_execution'] and completion['statistics'] == expected
    assert sum(e['updates'] for e in schedule) == binding['total_updates']
    history = read(folder / 'history.json')
    assert [e['epoch'] for e in history] == [1, 2]
    assert all(e['val_n'] == len(modelval) for e in history)
    assert all(e['skipped_no_supervision_batches'] == 0 for e in history)
    assert read(folder / 'answer_vocab.json') == read(config['data']['sealed_answer_vocab'])
    digests = {name: sha(folder / name) for name in ('last.pt', 'optimizer.pt')}
    assert digests['last.pt'] == completion['last_sha256']
    assert digests['optimizer.pt'] == completion['optimizer_sha256']
    report = {'passed': True, 'fold': fold, 'scope': 'Execution artifacts only; no inference or independent clinical/VQA metric validation',
              'statistics': expected, 'checkpoint_sha256': digests,
              'elapsed_seconds': completion['elapsed_seconds'],
              'independent_quality_audit_pending': True, 'reserved_role_inference': False}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))

if __name__ == '__main__':
    main(sys.argv[1])
