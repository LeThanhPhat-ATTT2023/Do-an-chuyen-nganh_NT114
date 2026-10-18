"""Experiment grid: expand axes, run cells sequentially (resumable), summarize.

Grid YAML: ``base_config``, ``output_root``, ``name_template``, ``overrides`` (applied to every
cell) and ``axes``: {axis: {label: {dotted.key: value}}}. A cell whose ``attack.name`` is ``none``
gets ``attack.malicious_ratio = 0`` and is deduplicated. A cell value that is a dict replaces the
whole sub-dict at that key (``defense.aggregator: {name: raguard, params: {...}}``).
"""

from __future__ import annotations

import csv
import fnmatch
import itertools
import json
import traceback
from pathlib import Path
from typing import Any

from graphslm_ids.offline.federated.config import apply_overrides, load_fl_config, unknown_keys, yaml_load
from graphslm_ids.offline.federated.context import data_key, load_context
from graphslm_ids.offline.federated.simulation import run_experiment

IMPACT_METRICS = ("final_test_macro_f1", "final_test_macro_f1_clean")
# Labels that describe the attack itself (``scenario``: when it starts, what it knows): they differ
# between a cell and its no-attack reference.
ATTACK_LABELS = ("label_attack", "label_malicious_ratio", "label_placement", "label_scenario")
_SORT_LABELS = ("split", "model", "partition", "seed")


def expand_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    axes = grid.get("axes") or {}
    names = list(axes)
    template = grid.get("name_template") or "__".join("{" + n + "}" for n in names)
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    for combo in itertools.product(*[list((axes[n] or {}).items()) for n in names]):
        labels = {n: str(label) for n, (label, _ov) in zip(names, combo)}
        overrides: dict[str, Any] = {}
        for _label, ov in combo:
            overrides.update(ov or {})
        if str(overrides.get("attack.name", "")).lower() == "none":
            overrides["attack.malicious_ratio"] = 0.0
            if "malicious_ratio" in labels:
                labels["malicious_ratio"] = "0"
        name = template.format(**labels)
        if name in seen:
            continue
        seen.add(name)
        cells.append({"name": name, "labels": labels, "overrides": overrides})
    # Consecutive cells share a data context and a client cache.
    cells.sort(key=lambda c: tuple(str(c["labels"].get(k, "")) for k in _SORT_LABELS))
    return cells


def run_grid(
    grid_path: str | Path,
    *,
    device: str | None = None,
    dry_run: bool = False,
    only: str | None = None,
    fail_fast: bool = False,
) -> list[dict[str, str]]:
    grid = yaml_load(Path(grid_path).read_text(encoding="utf-8")) or {}
    base = load_fl_config(grid["base_config"])
    bad = unknown_keys(base)
    if bad:
        raise ValueError(f"unknown keys in base config {grid['base_config']}: {bad}")
    root = Path(grid.get("output_root", "outputs/fl/matrix"))
    common = dict(grid.get("overrides") or {})
    if device:
        common["device"] = device
    cells = expand_grid(grid)
    # A config typo must raise before any cell trains, whatever --only selects.
    for cell in cells:
        bad = unknown_keys(apply_overrides(base, {**common, **cell["overrides"]}))
        if bad:
            raise ValueError(f"unknown config keys in cell {cell['name']!r}: {bad}")
    if only:
        cells = [c for c in cells if fnmatch.fnmatch(c["name"], only)]
    results: list[dict[str, str]] = []
    ctx = None
    for i, cell in enumerate(cells, start=1):
        out = root / cell["name"]
        if (out / "fl_summary.json").exists():
            results.append({"name": cell["name"], "status": "skipped"})
            continue
        if dry_run:
            print(f"[matrix] {i}/{len(cells)} {cell['name']}")
            results.append({"name": cell["name"], "status": "planned"})
            continue
        cfg = apply_overrides(
            base,
            {
                **common,
                **cell["overrides"],
                "output.dir": str(out),
                "experiment.name": cell["name"],
                "experiment.labels": cell["labels"],
            },
        )
        print(f"[matrix] {i}/{len(cells)} {cell['name']}", flush=True)
        try:
            if ctx is None or ctx.data_key != data_key(cfg):
                ctx = load_context(cfg)
            run_experiment(cfg, out_dir=out, context=ctx)
            (out / "error.txt").unlink(missing_ok=True)
            results.append({"name": cell["name"], "status": "done"})
        except Exception as exc:
            out.mkdir(parents=True, exist_ok=True)
            (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            first = (f"{type(exc).__name__}: {exc}".splitlines() or [type(exc).__name__])[0]
            print(f"[matrix] FAILED {cell['name']}: {first}", flush=True)
            results.append({"name": cell["name"], "status": "failed"})
            if fail_fast:
                raise
    return results


def _get(d: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": summary.get("experiment"),
        "mode": summary.get("mode"),
        "model": summary.get("model"),
        "attack": summary.get("attack"),
        "aggregator": summary.get("aggregator"),
    }
    row.update({f"label_{k}": v for k, v in (summary.get("labels") or {}).items()})
    if summary.get("mode") == "local_only":
        for metric in ("test_macro_f1", "test_macro_f1_clean"):
            row[f"final_{metric}"] = _get(summary, "aggregate", metric, "mean")
        for stat in ("std", "min", "max"):
            row[f"{stat}_test_macro_f1"] = _get(summary, "aggregate", "test_macro_f1", stat)
        row["wall_seconds"] = _get(summary, "cost", "wall_seconds")
        row["peak_rss_gb"] = _get(summary, "cost", "peak_rss_gb")
        return row
    row.update(
        {
            "final_test_macro_f1": _get(summary, "final", "test", "macro_f1"),
            "final_test_macro_f1_clean": _get(summary, "final", "test", "clean", "macro_f1"),
            "best_test_macro_f1": _get(summary, "best", "test", "macro_f1"),
            "best_test_macro_f1_clean": _get(summary, "best", "test", "clean", "macro_f1"),
            "final_test_benign_far": _get(summary, "final", "test", "benign_false_alarm_rate"),
            "final_test_attack_detection": _get(summary, "final", "test", "attack_detection_rate"),
        }
    )
    for family, recall in (_get(summary, "final", "test", "family_recall") or {}).items():
        row[f"final_test_recall_{family}"] = recall
    det = summary.get("detection") or {}
    benign_rounds = int(det.get("fp") or 0) + int(det.get("tn") or 0)
    row.update(
        {
            "detect_tpr": det.get("tpr"),
            "benign_false_rejection": det.get("fpr"),
            "downweight_rate_benign": (int(det.get("downweighted_benign") or 0) / benign_rounds) if benign_rounds else None,
            "malicious_acceptance": det.get("malicious_acceptance_rate"),
            "convergence_round": summary.get("convergence_round"),
            "mean_round_seconds": _get(summary, "cost", "mean_round_seconds"),
            "wall_seconds": _get(summary, "cost", "wall_seconds"),
            "peak_vram_gb": _get(summary, "cost", "peak_vram_gb"),
            "peak_rss_gb": _get(summary, "cost", "peak_rss_gb"),
            "bytes_up": _get(summary, "cost", "bytes_up"),
            "bytes_down": _get(summary, "cost", "bytes_down"),
        }
    )
    return row


def add_attack_impact(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """impact = metric(no-attack reference) - metric(cell).

    The reference of a cell is the ``fl`` / ``attack == none`` row whose labels equal the cell's
    on everything except ``ATTACK_LABELS`` (the attack, its ratio, placement and scenario), so a
    P1/P2 cell finds its P0 reference when a parent folder of both grid roots is summarized.
    """

    def key(r: dict[str, Any]) -> tuple:
        return tuple(sorted((k, str(v)) for k, v in r.items() if k.startswith("label_") and k not in ATTACK_LABELS))

    reference = {key(r): r for r in rows if r.get("attack") == "none" and r.get("mode") == "fl"}
    for r in rows:
        base = reference.get(key(r)) if r.get("mode") == "fl" else None
        for metric in IMPACT_METRICS:
            ok = base is not None and base.get(metric) is not None and r.get(metric) is not None
            r[f"impact_{metric}"] = (base[metric] - r[metric]) if ok else None
    return rows


def summarize(root: str | Path) -> list[dict[str, Any]]:
    rows = [
        summary_row(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(Path(root).rglob("fl_summary.json"))
    ]
    return add_attack_impact(rows)


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def pivot_markdown(
    rows: list[dict[str, Any]],
    value: str = "final_test_macro_f1",
    row_keys: tuple[str, ...] = ("label_attack", "label_malicious_ratio"),
    col_key: str = "label_defense",
    group_key: str = "label_partition",
) -> str:
    """One markdown table of ``value`` per ``group_key`` value, columns ``col_key``.

    A table row is keyed by every ``label_*`` column present in ``rows`` except ``col_key`` and
    ``group_key`` (``row_keys`` that are present first, the rest sorted), so runs that differ in
    any label never share a cell; a label a run lacks (another grid root) is blank. Two runs with
    identical labels would share a cell: ``ValueError``.
    """
    present = {k for r in rows for k in r if k.startswith("label_") or k in row_keys} - {col_key, group_key}
    keys_order = [k for k in row_keys if k in present] + sorted(present.difference(row_keys))

    def label(r: dict[str, Any], key: str) -> str:
        return "" if r.get(key) is None else str(r.get(key))

    lines: list[str] = []
    for group in sorted({str(r.get(group_key)) for r in rows}):
        sub = [r for r in rows if str(r.get(group_key)) == group]
        cols = sorted({str(r.get(col_key)) for r in sub})
        cell: dict[tuple[tuple[str, ...], str], Any] = {}
        owner: dict[tuple[tuple[str, ...], str], Any] = {}
        for r in sub:
            at = (tuple(label(r, k) for k in keys_order), str(r.get(col_key)))
            if at in owner:
                raise ValueError(
                    f"runs {owner[at]!r} and {r.get('name')!r} have identical labels {dict(zip(keys_order, at[0]))} "
                    f"({group_key}={group}, {col_key}={at[1]}); they would share one pivot cell"
                )
            owner[at] = r.get("name")
            cell[at] = r.get(value)
        keys = sorted({rk for rk, _c in cell})
        lines.append(f"### {group_key.removeprefix('label_')} = {group} - {value}")
        lines.append("")
        lines.append("| " + " | ".join([*(k.removeprefix("label_") for k in keys_order), *cols]) + " |")
        lines.append("|" + "---|" * (len(keys_order) + len(cols)))
        for rk in keys:
            lines.append("| " + " | ".join([*rk, *(_fmt(cell.get((rk, c))) for c in cols)]) + " |")
        lines.append("")
    return "\n".join(lines)
