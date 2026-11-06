"""Account explicitly for cached decoder errors; preserve failed v3_1 source."""
import json
from pathlib import Path
from experiments import audit_fracatlas_upstream_neighbours_v3_1 as core


def valid_rows():
    records = [json.loads(s) for s in core.SOURCE.read_text().splitlines()]
    selected = [r for r in records if any(x in r['path'] for x in ('\\fracatlas\\', '\\roco\\', '\\pmc_vqa\\'))]
    invalid = [r for r in selected if 'phash' not in r]
    assert all('error' in r for r in invalid)
    good = [r for r in selected if 'phash' in r]
    assert all(len(r['phash']) == 16 for r in good)
    assert len({r['path'] for r in selected}) == len(selected)
    # Every sealed fold0 role image must be covered, even if source corpus has errors.
    covered = {r['path'].split('\\')[-1] for r in good if '\\fracatlas\\' in r['path']}
    bindings = {}
    for role in ('train', 'model_val', 'calibration', 'test'):
        p = core.ROOT/f'.evaluation_runtime/data/fracatlas_component_cv_v3/fold_0/vqa_{role}.json'
        entries = json.loads(p.read_text())
        assert isinstance(entries, list)
        names = {r['image'].replace('\\', '/').split('/')[-1] for r in entries}
        assert names <= covered, (role, names-covered)
        bindings[role] = {'sha256': core.sha(p), 'unique_images': len(names)}
    valid_rows.coverage = {'cached_decoder_errors_excluded': invalid,
        'error_count': len(invalid), 'sealed_fold0_role_coverage': bindings,
        'all_sealed_fold0_images_have_valid_cached_fingerprints': True,
        'wrapper_sha256': core.sha(Path(__file__).resolve()),
        'failed_v3_1': 'Assertion before output/model: source cache contains decoder-error records; original source preserved'}
    return ([r for r in good if '\\fracatlas\\' in r['path']],
            [r for r in good if any(x in r['path'] for x in ('\\roco\\', '\\pmc_vqa\\'))])


if __name__ == '__main__':
    core.OUT = core.ROOT/'refactor_v2/reports/fracatlas_upstream_neighbours_v3_2'
    core.rows = valid_rows
    core.main()
    if not (core.OUT/'cache_coverage.json').exists():
        core.dump(core.OUT/'cache_coverage.json', valid_rows.coverage)
    else:
        assert json.loads((core.OUT/'cache_coverage.json').read_text()) == valid_rows.coverage
