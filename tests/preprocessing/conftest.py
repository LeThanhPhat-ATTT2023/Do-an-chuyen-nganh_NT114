"""Shared fixtures for v3 unit tests.

These fixtures keep every v3 test self-contained:

* No reliance on the real ``data/raw/14gb`` PCAPs (those are ~5 GB).
* No reliance on the real ``data/mitre/enterprise-attack.json`` (~30 MB).
* All deterministic so flaky-test-bisect isn't needed.

Fixture inventory:

* ``tiny_payloads`` — 10 hand-picked synthetic byte payloads spanning HTTP
  request/response, raw binary, ASCII, and the empty payload edge case.
* ``tiny_flows_df`` — 20 flow rows across 5 attack classes with
  linearly-spaced ``flow_start_ts``.
* ``tiny_packets_df`` — ~50 packet rows linked to ``tiny_flows_df`` via
  ``flow_id``.
* ``mitre_dir_fixture`` — temp dir holding a mock ``enterprise-attack.json``
  + ``mitre_techniques.csv`` + ``technique_tactic_edges.csv``.
* ``class_technique_fixture_path`` — temp CSV mapping fixture classes to
  techniques.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_payloads() -> list[bytes]:
    """Ten synthetic payloads covering the major byte-distribution regimes.

    Index 0 is the empty payload (an important edge case for every module),
    index 1 is plain ASCII (text tokens fire), index 2 is binary garbage
    (only byte n-grams fire), and so on.
    """
    return [
        b"",
        b"hello world this is plain text content",
        bytes(range(256)),  # full byte range, low printable ratio
        b"GET /index.html?id=1 HTTP/1.1\r\nHost: example.com\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html></html>",
        b"GET /?id=1' OR 1=1-- HTTP/1.1\r\nHost: target\r\n\r\n",
        b"<script>alert('xss')</script>",
        b"\x00" * 64,  # all zeros — printable ratio = 0
        b"information_schema UNION SELECT * FROM users",
        b"POST /upload HTTP/1.1\r\n\r\neval(base64_decode('cm0gLXJm'));",
    ]


@pytest.fixture
def tiny_flows_df() -> pd.DataFrame:
    """20 flows across 5 classes, linearly-spaced start times.

    Designed so the temporal split is unambiguous: timestamps within a class
    are strictly increasing, so the train/val/test cut points are clearly
    defined.
    """
    rows = []
    classes = ["Benign", "SqlInjection", "XSS", "Recon-PortScan", "DDoS"]
    flow_id = 0
    for cls_idx, cls in enumerate(classes):
        for i in range(4):  # 4 flows per class * 5 classes = 20 flows
            rows.append(
                {
                    "flow_id": f"f{flow_id:03d}",
                    "label": cls,
                    # Spread ts within class across [1000, 2000] so temporal
                    # cuts produce non-trivial buckets.
                    "flow_start_ts": 1000.0 + cls_idx * 100 + i * 25.0,
                    "src_ip": f"10.0.0.{cls_idx}",
                    "dst_ip": f"10.0.1.{i}",
                    "src_port": 1024 + flow_id,
                    "dst_port": 80 + (flow_id % 5),
                    "proto": 0,  # TCP
                }
            )
            flow_id += 1
    return pd.DataFrame(rows).set_index("flow_id")


@pytest.fixture
def tiny_packets_df(tiny_flows_df: pd.DataFrame) -> pd.DataFrame:
    """~50 packets linked to ``tiny_flows_df`` flows.

    Each packet carries a 256-byte uint8 payload buffer (matching v2's
    fixed-length packet feature layout). Payloads are deterministic functions
    of the flow_id + offset so tests can rely on byte values without
    reseeding RNGs.
    """
    rng = np.random.default_rng(42)
    rows = []
    pkt_idx = 0
    flow_ids = list(tiny_flows_df.index)
    for fid in flow_ids:
        # 2-3 packets per flow; minimum 2 so next_packet links exist.
        n_pkts = int(rng.integers(2, 4))
        ts0 = float(tiny_flows_df.loc[fid, "flow_start_ts"])
        label = str(tiny_flows_df.loc[fid, "label"])
        for j in range(n_pkts):
            # 256-byte deterministic payload — values depend on (flow, offset)
            # so duplicate-tokenisation tests can still find collisions.
            payload = (
                np.frombuffer(
                    f"{fid}:{j}:".encode("utf-8").ljust(256, b"X")[:256], dtype=np.uint8
                )
                .copy()
            )
            rows.append(
                {
                    "ts": ts0 + 0.01 * j,
                    "flow_id": fid,
                    "payload": payload,
                    "payload_len": 256,
                    "label": label,
                }
            )
            pkt_idx += 1
    return pd.DataFrame(rows)


@pytest.fixture
def mitre_dir_fixture(tmp_path: Path) -> Path:
    """Temp ``data/mitre``-shaped directory with stub STIX + CSVs.

    The STIX JSON has three attack-pattern objects, each with literal
    ``<code>`` blocks the ProcedureMatcher should pick up. Patterns are
    chosen to be unlikely-to-collide so positive/negative match tests are
    unambiguous.
    """
    mitre_dir = tmp_path / "mitre"
    mitre_dir.mkdir(parents=True, exist_ok=True)

    stix = {
        "type": "bundle",
        "id": "bundle--mock",
        "objects": [
            {
                "type": "attack-pattern",
                "id": "attack-pattern--1",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1190"}
                ],
                "description": (
                    "Adversaries inject payloads such as <code>UNION SELECT</code> "
                    "into web applications, often referencing "
                    "<code>information_schema</code>."
                ),
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--2",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1059"}
                ],
                "description": (
                    "Adversaries run `cat /etc/passwd` or invoke "
                    "<code>cmd.exe</code> for command execution."
                ),
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--3",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1046"}
                ],
                # No literal patterns — exercises the no-pattern branch.
                "description": "Adversaries scan network services.",
            },
        ],
    }
    stix_path = mitre_dir / "enterprise-attack.json"
    stix_path.write_text(json.dumps(stix), encoding="utf-8")

    # Minimal techniques.csv (only columns downstream code touches).
    techs = pd.DataFrame(
        [
            {"technique_id": "T1190", "name": "Exploit Public-Facing", "is_subtechnique": False},
            {"technique_id": "T1059", "name": "Command and Scripting", "is_subtechnique": False},
            {"technique_id": "T1046", "name": "Network Service Discovery", "is_subtechnique": False},
            {"technique_id": "T1059.004", "name": "Unix Shell", "is_subtechnique": True},
            {"technique_id": "T1059.007", "name": "JavaScript", "is_subtechnique": True},
        ]
    )
    techs.to_csv(mitre_dir / "mitre_techniques.csv", index=False)

    # Minimal technique→tactic edges CSV.
    tt = pd.DataFrame(
        [
            {"technique_id": "T1190", "tactic_id": "TA0001"},
            {"technique_id": "T1059", "tactic_id": "TA0002"},
            {"technique_id": "T1046", "tactic_id": "TA0007"},
        ]
    )
    tt.to_csv(mitre_dir / "technique_tactic_edges.csv", index=False)

    return mitre_dir


@pytest.fixture
def class_technique_fixture_path(tmp_path: Path) -> Path:
    """Minimal class→technique map CSV for the fixture's 5 classes."""
    df = pd.DataFrame(
        [
            {"class": "Benign", "technique": "", "weight": 0.0, "note": "no attack"},
            {"class": "SqlInjection", "technique": "T1190", "weight": 1.0, "note": ""},
            {"class": "XSS", "technique": "T1190", "weight": 0.7, "note": ""},
            {"class": "Recon-PortScan", "technique": "T1046", "weight": 1.0, "note": ""},
            {"class": "DDoS", "technique": "T1498", "weight": 1.0, "note": ""},
        ]
    )
    path = tmp_path / "class_technique_map.csv"
    df.to_csv(path, index=False)
    return path
