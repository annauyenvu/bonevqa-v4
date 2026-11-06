"""Isolate the installed MPS RNG replay behavior without model training."""
import json
from pathlib import Path

import torch


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("This diagnostic requires MPS, not a CPU fallback")
    x = torch.ones(1024, device="mps")
    torch.mps.manual_seed(42)
    torch.nn.functional.dropout(x, p=0.1, training=True)
    state = torch.mps.get_rng_state()
    expected = torch.nn.functional.dropout(x, p=0.1, training=True)
    torch.mps.set_rng_state(state)
    restored = torch.nn.functional.dropout(x, p=0.1, training=True)
    torch.mps.manual_seed(123)
    first = torch.nn.functional.dropout(x, p=0.1, training=True)
    torch.mps.manual_seed(123)
    second = torch.nn.functional.dropout(x, p=0.1, training=True)
    report = {
        "torch_version": torch.__version__, "device": "mps", "samples": 1024,
        "rng_state_bytes": state.numel(),
        "state_restore_dropout_equal": torch.equal(expected, restored),
        "state_restore_differing_elements": int((expected != restored).sum()),
        "manual_seed_dropout_equal": torch.equal(first, second),
        "scope": "installed runtime only; epoch-boundary seed workaround, not mid-epoch replay",
    }
    Path("refactor_v2/reports/mps_rng_replay_probe_v3.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["manual_seed_dropout_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
