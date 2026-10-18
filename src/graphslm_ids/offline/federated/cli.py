"""Run one FL poisoning experiment.

    python -m graphslm_ids.offline.federated.cli --config configs/fl/fl_base.yaml \
        --device cuda --set attack.name=alie --set attack.malicious_ratio=0.2
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from graphslm_ids.offline.federated.config import apply_overrides, load_fl_config, parse_set_args
from graphslm_ids.offline.federated.records import to_jsonable
from graphslm_ids.offline.federated.simulation import VALID_MODES, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one FL poisoning experiment.")
    parser.add_argument("--config", required=True, help="FL YAML config (configs/fl/*.yaml)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted override, repeatable")
    parser.add_argument("--device", help="cpu | cuda | auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--mode", choices=list(VALID_MODES))
    parser.add_argument("--rounds", type=int)
    return parser


def headline(summary: dict[str, Any]) -> dict[str, Any]:
    final = (summary.get("final") or {}).get("test") or {}
    return {
        "experiment": summary.get("experiment"),
        "mode": summary.get("mode"),
        "model": summary.get("model"),
        "attack": summary.get("attack"),
        "aggregator": summary.get("aggregator"),
        "test_macro_f1": final.get("macro_f1"),
        "test_macro_f1_clean": (final.get("clean") or {}).get("macro_f1"),
        "detection": summary.get("detection"),
        "aggregate": summary.get("aggregate"),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = parse_set_args(args.set)
    if args.device:
        overrides["device"] = args.device
    if args.output_dir:
        overrides["output.dir"] = args.output_dir
    if args.mode:
        overrides["federated.mode"] = args.mode
    if args.rounds is not None:
        overrides["federated.rounds"] = args.rounds
    cfg = apply_overrides(load_fl_config(args.config), overrides)
    summary = run_experiment(cfg)
    print(json.dumps(to_jsonable(headline(summary)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
