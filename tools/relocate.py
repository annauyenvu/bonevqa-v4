"""Rewrite the absolute project root used during the original runs to this clone's location.

The experiment scripts were run from /Users/uyen/Downloads/BONEVQA and keep that path as a
literal, so that the code in this repository is byte-identical to the code that produced the
reported numbers. Run this once after cloning:

    python tools/relocate.py            # rewrite to the directory that contains tools/
    python tools/relocate.py --check    # only count the files that would change

Only text files (.py, .sh, .yaml, .yml, .json, .md, .txt) under refactor_v2/, exp4_multiseed/
and mandatory_experiments_20261011/ are touched. Use git to see or revert the changes.
"""
import argparse
from pathlib import Path

OLD = "/Users/uyen/Downloads/BONEVQA"
SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".txt"}
DIRS = ("refactor_v2", "exp4_multiseed", "mandatory_experiments_20261011")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    new = str(Path(a.root).resolve())
    changed = 0
    for d in DIRS:
        for p in (Path(new) / d).rglob("*"):
            if not p.is_file() or p.suffix not in SUFFIXES:
                continue
            try:
                s = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if OLD in s:
                changed += 1
                if not a.check:
                    p.write_text(s.replace(OLD, new), encoding="utf-8")
    print(f"{'would change' if a.check else 'changed'} {changed} files: {OLD} -> {new}")


if __name__ == "__main__":
    main()
