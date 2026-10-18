"""Saved client deltas and offline detection replay.

A live FedAvg run with attackers can save every round's ``(N, D)`` client deltas
(``output.save_deltas``). Any defense is then scored on detection (TPR, benign false-rejection)
by stepping one fresh server through the saved rounds, without retraining. Macro-F1 under
attack still needs a full run per defense. This module imports no model/graph/trainer code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from graphslm_ids.offline.federated.defense import build_defense
from graphslm_ids.offline.federated.metrics import detection_counts, detection_rates, merge_counts
from graphslm_ids.offline.federated.params import ParamLayout
from graphslm_ids.offline.federated.records import write_json
from graphslm_ids.offline.federated.server import ClientMessage, FLServer

FLOAT16_MAX = 65504.0
_DTYPES = {"float16": torch.float16, "float32": torch.float32}


def save_round_deltas(
    out_dir: Path,
    round_idx: int,
    client_ids: list[int],
    deltas: torch.Tensor,
    num_samples: torch.Tensor,
    malicious: list[int],
    dtype: str = "float16",
) -> str:
    """Write ``round_{r:03d}.pt`` on CPU; returns the dtype used (float32 if float16 would overflow)."""
    if dtype not in _DTYPES:
        raise ValueError(f"unknown delta dtype {dtype!r}; expected one of {sorted(_DTYPES)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cpu = deltas.detach().to("cpu", torch.float32)
    used = dtype
    if used == "float16" and cpu.numel() and float(cpu.abs().max()) > FLOAT16_MAX:
        used = "float32"
    torch.save(
        {
            "round": int(round_idx),
            "client_ids": [int(c) for c in client_ids],
            "deltas": cpu.to(_DTYPES[used]),
            "num_samples": num_samples.detach().to("cpu", torch.float32),
            "malicious": [int(c) for c in malicious],
        },
        out_dir / f"round_{int(round_idx):03d}.pt",
    )
    return used


def load_round_deltas(path: str | Path) -> dict[str, Any]:
    rec = torch.load(Path(path), map_location="cpu", weights_only=False)
    rec["deltas"] = rec["deltas"].to(torch.float32)
    return rec


def write_delta_meta(
    out_dir: Path,
    layout: ParamLayout,
    *,
    weighting: str,
    server_lr: float,
    malicious_clients: list[int],
    rounds: int,
    assumed_malicious_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    write_json(
        Path(out_dir) / "meta.json",
        {
            "numel": int(layout.numel),
            "groups": layout.groups_json(),
            "weighting": weighting,
            "server_lr": float(server_lr),
            "malicious_clients": [int(c) for c in malicious_clients],
            "rounds": int(rounds),
            "assumed_malicious_ratio": float(assumed_malicious_ratio),
            "seed": int(seed),
        },
    )


def replay(
    run_dir: str | Path,
    defense_cfg: dict,
    *,
    assumed_malicious_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Score ``defense_cfg`` on the deltas saved under ``run_dir/deltas``."""
    deltas_dir = Path(run_dir) / "deltas"
    meta = json.loads((deltas_dir / "meta.json").read_text(encoding="utf-8"))
    layout = ParamLayout.from_groups({g: [tuple(s) for s in spans] for g, spans in meta["groups"].items()}, meta["numel"])
    aggregator = build_defense(defense_cfg, assumed_malicious_ratio=assumed_malicious_ratio, seed=seed)
    server = FLServer(
        global_vec=torch.zeros(int(meta["numel"])),
        aggregator=aggregator,
        layout=layout,
        weighting=meta["weighting"],
        server_lr=float(meta.get("server_lr", 1.0)),
    )
    rounds: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    for path in sorted(deltas_dir.glob("round_*.pt")):
        rec = load_round_deltas(path)
        messages = [
            ClientMessage(int(c), server.round_idx, rec["deltas"][i], int(round(float(rec["num_samples"][i]))))
            for i, c in enumerate(rec["client_ids"])
        ]
        report = server.step(messages)
        decisions = report.decision_map()
        counts = detection_counts(decisions, {int(c) for c in rec["malicious"]})
        totals = merge_counts(totals, counts)
        rounds.append({"round": int(rec["round"]), "decisions": decisions, "detection": counts})
    return {
        "defense": aggregator.name,
        "rounds": rounds,
        "detection": {**totals, **detection_rates(totals)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay saved client deltas through defenses.")
    parser.add_argument("--run", required=True, help="run directory containing deltas/")
    parser.add_argument("--defense", action="append", required=True, help="defense name (repeatable)")
    parser.add_argument(
        "--params-json",
        default=None,
        help='JSON object (or path to a JSON file) mapping defense name to its params, e.g. {"krum": {"f": 1}}',
    )
    parser.add_argument("--out", default=None, help="output path (default <run>/offline_eval.json)")
    args = parser.parse_args(argv)

    run = Path(args.run)
    params: dict[str, Any] = {}
    if args.params_json:
        text = args.params_json
        params = json.loads(Path(text).read_text(encoding="utf-8") if Path(text).is_file() else text)
    meta_path = run / "deltas" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    kwargs = {
        "assumed_malicious_ratio": float(meta.get("assumed_malicious_ratio", 0.2)),
        "seed": int(meta.get("seed", 42)),
    }
    result = {name: replay(run, {"name": name, "params": params.get(name) or {}}, **kwargs) for name in args.defense}
    write_json(Path(args.out) if args.out else run / "offline_eval.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
