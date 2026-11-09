"""Compare two result JSON files numerically. Strings that are file paths are ignored (they differ by clone location).
Usage: python compare_json.py <stored.json> <regenerated.json>   -> prints max |diff| and the number of numeric leaves."""
import json, sys


def leaves(x, p=''):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, f'{p}/{k}')
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from leaves(v, f'{p}[{i}]')
    else: yield p, x


a = dict(leaves(json.load(open(sys.argv[1])))); b = dict(leaves(json.load(open(sys.argv[2]))))
num = [k for k, v in a.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
missing = [k for k in num if k not in b]
d = max([abs(a[k] - b[k]) for k in num if k in b and isinstance(b[k], (int, float))] or [0.0])
other = [k for k, v in a.items() if k not in num and isinstance(v, (str, bool)) and k in b and b[k] != v and not (isinstance(v, str) and '/' in v)]
print(f'numeric_leaves={len(num)} missing={len(missing)} max_abs_diff={d:.3g} nonpath_string_mismatch={len(other)}')
sys.exit(0 if not missing and d < 1e-9 and not other else 1)
