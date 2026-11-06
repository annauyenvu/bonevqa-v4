"""Inventory sparse author-code snapshots; no model imports or training."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'FAVP': ('favp_author_b43941ba', 'b43941bac4774b796fd8650a263ea32c220e6aba'),
    'LocalizationLens': ('lens_author_22a94007', '22a940079c8a6b9d94994c3b950ad4b28e7b253c'),
}


def main():
    result = {'source_only': True, 'model_execution': False, 'training_release': False, 'sources': {}}
    for name, (directory, expected) in SOURCES.items():
        source = ROOT / 'refactor_v2/references' / directory
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        assert commit == expected
        assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
        tracked = subprocess.check_output(['git', '-C', str(source), 'ls-files'], text=True).splitlines()
        present = [source / p for p in tracked if (source / p).is_file()]
        skipped = [p for p in tracked if not (source / p).is_file()]
        result['sources'][name] = {
            'commit': commit, 'root': str(source), 'clean': True,
            'sparse_paths': subprocess.check_output(['git', '-C', str(source), 'sparse-checkout', 'list'], text=True).splitlines(),
            'checked_out_files': len(present), 'not_checked_out_count': len(skipped),
            'not_checked_out_paths': skipped,
            'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in present},
            'top_level_license_files': [p.name for p in source.iterdir() if p.is_file() and p.name.lower().startswith(('license', 'copying'))],
        }
    output = ROOT / 'refactor_v2/reports/vl_author_sources_v1'
    output.mkdir(exist_ok=False)
    with (output / 'source_manifest.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: {n: v for n, v in item.items() if n not in ('sha256', 'not_checked_out_paths')} for k, item in result['sources'].items()}, indent=2))


if __name__ == '__main__':
    main()
