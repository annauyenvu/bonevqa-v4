"""Resume the sealed pipeline after a documented pre-training technical fix."""
from common import *
import subprocess

setup(42)
verify_lock()
failure = read(ROOT / 'TECHNICAL_FAILURE.json')
assert 'confidence' not in failure.get('error', '') or failure
with (ROOT / 'logs/pipeline.log').open('a') as log:
    p = subprocess.Popen(
        [sys.executable, str(CODE / 'orchestrate.py')],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        env={**os.environ, 'PYTHONHASHSEED': '42'},
    )
write(ROOT / 'pipeline_pid.json', {
    'pid': p.pid,
    'python': sys.executable,
    'protocol_sha256': sha(P / 'LOCK.json'),
    'resume_reason': 'documented pre-training confidence-metadata amendment',
})
print('RESUMED', p.pid, sha(P / 'LOCK.json'), flush=True)
