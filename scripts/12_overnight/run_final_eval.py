"""Run `scripts/6_evaluate/evaluate_test_set.py`'s evaluation for a serial
stage-runner (fused waveform + demographics) checkpoint, without editing
that script.

`evaluate_test_set.py` only knows to load demographic features for a
checkpoint tagged `"cnn_combined"` — its `_DEMO_INPUT_TAGS` set is a
hardcoded one-element set. The `longrun`/`warmstart_tune` stage checkpoints
produced by `scripts/12_overnight/run_overnight.py` are also fused
(waveform + demographics, same `ECGConvNet(n_demo_features>0)`
architecture), so loading one under a plain `--tag cnn_serial` would make
`evaluate_test_set.py` build a demo-less model and fail (or silently
mis-score) when loading the checkpoint's state dict.

Since `scripts/6_evaluate/evaluate_test_set.py` is off-limits to edit here,
this wrapper imports it as a module (same pattern
`tests/conftest.py::load_script` uses) and adds the serial tag to its
`_DEMO_INPUT_TAGS` set at runtime before calling its evaluation entry point
— extending behaviour from the outside, not modifying the file.

It expects `checkpoints/<tag>_demo_encoder.joblib` to already exist (written
by `scripts/12_overnight/run_overnight.py`'s `load_shared_data`), matching
the path `evaluate_test_set.demo_features_for_tag` looks up for any tag in
`_DEMO_INPUT_TAGS`. Called automatically by the `final_eval` stage (only
when `longrun` or `warmstart_tune` produced a checkpoint that beat the
k-fold SHD mean); can also be run by hand:

Usage:
  .venv/bin/python -u scripts/12_overnight/run_final_eval.py \\
      --checkpoint checkpoints/serial/longrun_best.pt --tag cnn_serial
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_evaluate_test_set_module():
    path = PROJECT_ROOT / "scripts" / "6_evaluate" / "evaluate_test_set.py"
    spec = importlib.util.spec_from_file_location("evaluate_test_set", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_test_set"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="cnn_serial")
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()

    encoder_path = PROJECT_ROOT / "checkpoints" / f"{args.tag}_demo_encoder.joblib"
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"{encoder_path} is missing — scripts/12_overnight/run_overnight.py's shared-data "
            "loader should have written it (from the same encoder the night checkpoint's "
            "demographic features were built with) before this script is invoked."
        )

    m = load_evaluate_test_set_module()
    m._DEMO_INPUT_TAGS.add(args.tag)
    print(f"Extended evaluate_test_set._DEMO_INPUT_TAGS with {args.tag!r} -> {m._DEMO_INPUT_TAGS}", flush=True)

    config = m.build_run_config(args.checkpoint, args.tag, refit=args.refit)
    m.run_evaluation(config)


if __name__ == "__main__":
    main()
