from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MITRE ATT&CK node and edge CSV files from enterprise ATT&CK STIX JSON."
    )
    parser.add_argument(
        "--input-json",
        default="data/mitre/enterprise-attack.json",
        help="Path to enterprise-attack.json downloaded from MITRE CTI.",
    )
    parser.add_argument(
        "--techniques-csv",
        default="data/mitre/mitre_techniques.csv",
        help="Output CSV path for technique nodes.",
    )
    parser.add_argument(
        "--tactics-csv",
        default="data/mitre/mitre_tactics.csv",
        help="Output CSV path for tactic nodes.",
    )
    parser.add_argument(
        "--technique-tactic-edges-csv",
        default="data/mitre/mitre_technique_tactic_edges.csv",
        help="Output CSV path for technique->tactic edges.",
    )
    parser.add_argument(
        "--stats-json",
        default="data/mitre/mitre_export_stats.json",
        help="Output stats JSON path.",
    )
    return parser.parse_args()


def get_attack_external_id(stix_object: dict[str, Any], prefix: str) -> str | None:
    for ref in stix_object.get("external_references", []):
        source_name = str(ref.get("source_name", ""))
        external_id = str(ref.get("external_id", ""))
        if source_name in {"mitre-attack", "mitre-mobile-attack", "mitre-ics-attack"} and external_id.startswith(prefix):
            return external_id
    return None


def compact_text(value: Any) -> str:
    text = str(value or "")
    return " ".join(text.replace("\n", " ").split())


def unique_join(values: list[str]) -> str:
    cleaned = [v.strip() for v in values if str(v).strip()]
    unique_sorted = sorted(set(cleaned))
    return ";".join(unique_sorted)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    input_json = Path(args.input_json)
    techniques_csv = Path(args.techniques_csv)
    tactics_csv = Path(args.tactics_csv)
    edges_csv = Path(args.technique_tactic_edges_csv)
    stats_json = Path(args.stats_json)

    with input_json.open("r", encoding="utf-8") as handle:
        stix_bundle = json.load(handle)

    objects = stix_bundle.get("objects", [])

    tactic_rows: list[dict[str, Any]] = []
    for obj in objects:
        if obj.get("type") != "x-mitre-tactic":
            continue
        if obj.get("revoked") is True or obj.get("x_mitre_deprecated") is True:
            continue

        tactic_id = get_attack_external_id(obj, "TA")
        if not tactic_id:
            continue

        tactic_rows.append(
            {
                "tactic_id": tactic_id,
                "stix_id": obj.get("id", ""),
                "name": obj.get("name", ""),
                "shortname": obj.get("x_mitre_shortname", ""),
                "description": compact_text(obj.get("description", "")),
                "url": next(
                    (
                        ref.get("url", "")
                        for ref in obj.get("external_references", [])
                        if ref.get("external_id", "") == tactic_id
                    ),
                    "",
                ),
            }
        )

    tactic_rows.sort(key=lambda row: row["tactic_id"])

    technique_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") is True or obj.get("x_mitre_deprecated") is True:
            continue

        technique_id = get_attack_external_id(obj, "T")
        if not technique_id:
            continue

        kill_chain_phases = obj.get("kill_chain_phases", [])
        technique_tactics: list[str] = []
        for phase in kill_chain_phases:
            if phase.get("kill_chain_name") != "mitre-attack":
                continue
            phase_name = str(phase.get("phase_name", "")).strip()
            if not phase_name:
                continue
            technique_tactics.append(phase_name)
            edge_rows.append(
                {
                    "edge_id": f"{technique_id}::{phase_name}",
                    "technique_id": technique_id,
                    "tactic_shortname": phase_name,
                    "relation": "in_tactic",
                }
            )

        platforms = [str(p) for p in obj.get("x_mitre_platforms", [])]
        data_sources = [str(p) for p in obj.get("x_mitre_data_sources", [])]

        technique_rows.append(
            {
                "technique_id": technique_id,
                "stix_id": obj.get("id", ""),
                "name": obj.get("name", ""),
                "description": compact_text(obj.get("description", "")),
                "is_subtechnique": bool(obj.get("x_mitre_is_subtechnique", False)),
                "platforms": unique_join(platforms),
                "data_sources": unique_join(data_sources),
                "tactics": unique_join(technique_tactics),
                "url": next(
                    (
                        ref.get("url", "")
                        for ref in obj.get("external_references", [])
                        if ref.get("external_id", "") == technique_id
                    ),
                    "",
                ),
            }
        )

    technique_rows.sort(key=lambda row: row["technique_id"])
    edge_rows.sort(key=lambda row: (row["technique_id"], row["tactic_shortname"]))

    write_csv(
        techniques_csv,
        technique_rows,
        [
            "technique_id",
            "stix_id",
            "name",
            "description",
            "is_subtechnique",
            "platforms",
            "data_sources",
            "tactics",
            "url",
        ],
    )
    write_csv(
        tactics_csv,
        tactic_rows,
        ["tactic_id", "stix_id", "name", "shortname", "description", "url"],
    )
    write_csv(
        edges_csv,
        edge_rows,
        ["edge_id", "technique_id", "tactic_shortname", "relation"],
    )

    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_json": str(input_json),
        "outputs": {
            "techniques_csv": str(techniques_csv),
            "tactics_csv": str(tactics_csv),
            "technique_tactic_edges_csv": str(edges_csv),
        },
        "counts": {
            "techniques": len(technique_rows),
            "tactics": len(tactic_rows),
            "technique_tactic_edges": len(edge_rows),
        },
    }
    with stats_json.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(f"[OK] Techniques CSV: {techniques_csv}")
    print(f"[OK] Tactics CSV: {tactics_csv}")
    print(f"[OK] Technique->Tactic edges CSV: {edges_csv}")
    print(f"[OK] Export stats: {stats_json}")


if __name__ == "__main__":
    main()
