#!/usr/bin/env python3
from __future__ import annotations

import datetime
from pathlib import Path

from runtime import EXP4_CODE, LEGACY, PROJECT, ROOT, configure


def main() -> None:
    common, engine, specs = configure()
    common.setup(42)
    protocol = ROOT / "protocol"
    if (protocol / "LOCK.json").exists():
        raise RuntimeError("protocol already sealed")
    complete = common.read(ROOT / "roi_bank/COMPLETE.json")
    assert complete["roi_failed"] == 0
    audit = common.read(ROOT / "roi_bank/detector/audit.json")
    assert audit["expected_images"] == audit["processed_images"] and audit["roi_failed"] == 0
    assert common.sha(ROOT / "roi_bank/detector/index.json") == audit["index_sha256"]

    environment = common.environment()
    common.write(protocol / "environment.json", environment)
    (protocol / "environment.txt").write_text(__import__("json").dumps(environment, indent=2), encoding="utf-8")
    files = {}
    for path_text, digest in common.read(protocol / "input_hashes.json").items():
        path = Path(path_text)
        assert common.sha(path) == digest
        files[path_text] = digest
    for path in protocol.glob("*"):
        if path.is_file() and path.name != "LOCK.json":
            files[str(path.relative_to(ROOT))] = common.sha(path)
    for path in (ROOT / "code").glob("*.py"):
        files[str(path.relative_to(ROOT))] = common.sha(path)
    for path in EXP4_CODE.rglob("*.py"):
        files[str(path)] = common.sha(path)
    for path in (LEGACY / "deps/open_clip").rglob("*.py"):
        files[str(path)] = common.sha(path)
    for path in (ROOT / "roi_bank").glob("*.json"):
        files[str(path.relative_to(ROOT))] = common.sha(path)
    files["roi_bank/detector/index.json"] = common.sha(ROOT / "roi_bank/detector/index.json")
    files["roi_bank/detector/audit.json"] = common.sha(ROOT / "roi_bank/detector/audit.json")
    common.write(protocol / "LOCK.json", {"locked_at": datetime.datetime.now().isoformat(), "files": files})

    for seed in [42, 123, 456, 789, 2026]:
        for variant in ["none", "v", "p", "vp"]:
            dest = ROOT / "detector" / variant / f"seed_{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            common.write(dest / "config.json", engine.run_config("detector", variant, seed))
            (dest / "seed.txt").write_text(f"{seed}\n", encoding="utf-8")
            (dest / "environment.txt").write_text((protocol / "environment.txt").read_text(), encoding="utf-8")
            for split in specs:
                (dest / f"{split}_manifest_hash.txt").write_text(common.sha(protocol / f"{split}_manifest.json") + "\n")
    common.status("protocol", "sealed", protocol_sha256=common.sha(protocol / "LOCK.json"), primary_runs=20)
    print("SEALED", common.sha(protocol / "LOCK.json"), flush=True)


if __name__ == "__main__":
    main()

