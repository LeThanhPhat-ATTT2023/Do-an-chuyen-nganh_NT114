"""Sanity-gate audit for the v3 graph artifact.

Mirrors the gates defined in
``docs/superpowers/plans/2026-05-24-v3-smart-both-hybrid.md`` Section 10.

Exit code is 0 iff EVERY gate passes (or is reduced to a warning in
non-strict mode). Bracketed-range gates (flow counts, host counts, etc.) are
WARNINGS by default — full-scale 14 GB builds should pass them, but tiny
test fixtures (the 5-flow synthetic artifact used in CI) cannot. Pass
``--strict`` to promote them to failures when auditing the real build.

Usage:
    D:\\v\\nt114\\Scripts\\python.exe scripts/diagnostics/v3_artifact_audit.py \\
        --npz outputs/v3/graph.npz \\
        --meta outputs/v3/graph.meta.json \\
        --pmi outputs/v3/pmi_table.parquet \\
        --strict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


_TYPED_EVIDENCE_EDGES = [
    "evidence_injection_edge_index",
    "evidence_command_exec_edge_index",
    "evidence_file_upload_edge_index",
    "evidence_recon_edge_index",
    "evidence_c2_beacon_edge_index",
]


class Gate:
    """Single sanity-check result. ``severity`` decides exit-code contribution."""

    __slots__ = ("name", "ok", "msg", "severity")

    def __init__(self, name: str, ok: bool, msg: str, severity: str = "fail") -> None:
        # severity: "fail" → contributes to exit 1; "warn" → printed but ignored.
        self.name = name
        self.ok = ok
        self.msg = msg
        self.severity = severity

    def emit(self) -> None:
        glyph = "PASS" if self.ok else ("FAIL" if self.severity == "fail" else "WARN")
        print(f"[{glyph}] {self.name}: {self.msg}")


def _check_range(
    name: str,
    value: int,
    lo: int,
    hi: int,
    *,
    strict: bool,
) -> Gate:
    ok = lo <= value <= hi
    sev = "fail" if strict else "warn"
    return Gate(name, ok, f"{value} in [{lo}, {hi}]", severity=sev)


def _check_finite(name: str, arr: np.ndarray) -> Gate:
    if arr.size == 0:
        return Gate(name, True, "empty array (trivially finite)")
    bad = (~np.isfinite(arr)).sum()
    return Gate(name, bad == 0, f"{int(bad)} non-finite of {int(arr.size)} entries")


def audit(
    npz_path: Path,
    meta_path: Path,
    pmi_path: Path | None,
    strict: bool,
) -> list[Gate]:
    """Run the full v3 gate battery. Returns the list of Gate results."""
    gates: list[Gate] = []

    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Gate 1: artifact_version
    gates.append(
        Gate(
            "artifact_version",
            meta.get("artifact_version") == "v3",
            f"meta.artifact_version={meta.get('artifact_version')!r}",
        )
    )

    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {k: loaded[k] for k in loaded.files}

    # Node-count gates (warnings by default; promoted to fail under --strict).
    flow_x = arrays.get("flow_x")
    packet_x = arrays.get("packet_x")
    host_x = arrays.get("host_x")
    technique_x = arrays.get("technique_x")
    num_flows = int(flow_x.shape[0]) if flow_x is not None else 0
    num_packets = int(packet_x.shape[0]) if packet_x is not None else 0
    num_hosts = int(host_x.shape[0]) if host_x is not None else 0
    num_techniques = int(technique_x.shape[0]) if technique_x is not None else 0
    num_tactics = int(meta.get("num_tactics", 0))

    gates.append(_check_range("num_flows", num_flows, 150_000, 200_000, strict=strict))

    # num_packets <= 0.45 * num_packets_raw (control packets dropped).
    raw = int(meta.get("num_packets_raw", 0))
    if raw > 0:
        gates.append(
            Gate(
                "control_packets_dropped",
                num_packets <= int(raw * 0.45),
                f"num_packets={num_packets} <= 0.45 * num_packets_raw={raw} "
                f"({int(raw * 0.45)})",
                severity="fail" if strict else "warn",
            )
        )
    else:
        gates.append(
            Gate(
                "control_packets_dropped",
                True,
                "num_packets_raw not present in meta — gate skipped",
                severity="warn",
            )
        )

    gates.append(
        Gate(
            "num_techniques",
            num_techniques == 691,
            f"got {num_techniques}, expected 691",
            severity="fail" if strict else "warn",
        )
    )
    gates.append(
        Gate(
            "num_tactics",
            num_tactics == 14,
            f"got {num_tactics}, expected 14",
            severity="fail" if strict else "warn",
        )
    )
    gates.append(_check_range("num_hosts", num_hosts, 3_000, 15_000, strict=strict))

    # Active-technique count (degree > 0 from packets via typed-evidence edges).
    active_set: set[int] = set()
    for ek in _TYPED_EVIDENCE_EDGES:
        ei = arrays.get(ek)
        if ei is None or ei.size == 0:
            continue
        active_set.update(int(x) for x in np.unique(ei[1]))
    gates.append(
        _check_range("num_active_techniques", len(active_set), 9, 20, strict=strict)
    )

    # Per-typed-evidence edge count gate.
    for ek in _TYPED_EVIDENCE_EDGES:
        ei = arrays.get(ek)
        n = 0 if ei is None else int(ei.shape[1])
        # Empty allowed for tiny fixtures; range only enforced under --strict.
        if strict:
            ok = (n > 100) and (n < 10_000_000)
            msg = f"{n} edges, expected (100, 10M)"
            gates.append(Gate(f"edges.{ek}", ok, msg))
        else:
            gates.append(
                Gate(
                    f"edges.{ek}",
                    True,
                    f"{n} edges (range check skipped in non-strict mode)",
                    severity="warn",
                )
            )

    # No NaN / Inf in any node feature matrix. These are structural — always fail.
    if flow_x is not None:
        gates.append(_check_finite("flow_x_finite", flow_x))
    if packet_x is not None:
        gates.append(_check_finite("packet_x_finite", packet_x))
    if technique_x is not None:
        gates.append(_check_finite("technique_x_finite", technique_x))
    if host_x is not None:
        gates.append(_check_finite("host_x_finite", host_x))

    # has_subtechnique edges present.
    has_sub = arrays.get("has_subtechnique_edge_index")
    n_sub = 0 if has_sub is None else int(has_sub.shape[1])
    gates.append(Gate("has_subtechnique_edges", n_sub > 0, f"{n_sub} edges"))

    # Every flow node has degree > 0 via flow→contains→packet edges.
    contain = arrays.get("contain_edge_index")
    if contain is not None and contain.size > 0 and num_flows > 0:
        unique_flows = np.unique(contain[0])
        ok = unique_flows.size == num_flows
        gates.append(
            Gate(
                "flow_degree_gt_zero",
                ok,
                f"{unique_flows.size}/{num_flows} flows have contain edges",
            )
        )
    else:
        gates.append(
            Gate(
                "flow_degree_gt_zero",
                False,
                "contain_edge_index missing or empty",
            )
        )

    # PMI table — ≥ 5 tokens per technique family present.
    if pmi_path is not None and pmi_path.exists():
        try:
            import pandas as pd

            df = pd.read_parquet(pmi_path)
            if "family" not in df.columns:
                gates.append(
                    Gate(
                        "pmi_per_family",
                        False,
                        f"pmi_table missing 'family' column (cols={list(df.columns)})",
                    )
                )
            else:
                counts = df.groupby("family").size()
                bad = counts[counts < 5]
                gates.append(
                    Gate(
                        "pmi_per_family",
                        bad.empty,
                        f"families with <5 tokens: {dict(bad)}",
                    )
                )
        except Exception as exc:  # noqa: BLE001 — surface any parquet error
            gates.append(
                Gate("pmi_per_family", False, f"could not read pmi_table: {exc!r}")
            )
    else:
        gates.append(
            Gate(
                "pmi_per_family",
                True,
                "pmi_table path not supplied — gate skipped",
                severity="warn",
            )
        )

    return gates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, type=Path, help="path to graph.npz")
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="path to graph.meta.json; defaults to <npz>.meta.json",
    )
    parser.add_argument(
        "--pmi",
        type=Path,
        default=None,
        help="path to pmi_table.parquet (optional)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="promote bracketed-range checks from warnings to failures",
    )
    args = parser.parse_args()

    meta = args.meta if args.meta is not None else args.npz.with_suffix(".meta.json")

    gates = audit(args.npz, meta, args.pmi, args.strict)
    for g in gates:
        g.emit()

    failures = [g for g in gates if not g.ok and g.severity == "fail"]
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
