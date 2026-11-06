#!/usr/bin/env python3
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from runtime import configure


def main() -> None:
    common, engine, specs = configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train"])
    parser.add_argument("--variant", choices=["none", "v", "p", "vp"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    try:
        engine.train_run("detector", args.variant, args.seed, args.dest)
        qc = common.read(args.dest / "QC.json")
        qc.update({
            "test_qa": specs["test"][0],
            "closed_qa": specs["test"][2],
            "test_images": specs["test"][1],
            "component_clean": True,
            "variant_definition": {
                "use_visual_prompt": "v" in args.variant,
                "use_lens": False,
                "use_latent_prompt": "p" in args.variant,
            },
        })
        common.write(args.dest / "QC.json", qc)
    except Exception as exc:
        common.write(args.dest / "QC.json", {
            "valid": False,
            "technical_failure": repr(exc),
            "traceback": traceback.format_exc(),
            "seed": args.seed,
            "regime": "detector",
            "variant": args.variant,
        })
        common.status("technical_failure", "error", run=str(args.dest), error=repr(exc))
        raise


if __name__ == "__main__":
    main()
