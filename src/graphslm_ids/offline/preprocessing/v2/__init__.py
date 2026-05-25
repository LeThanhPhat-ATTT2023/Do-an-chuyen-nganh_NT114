"""v2 graph-construction pipeline (evidence-grounded heterogeneous threat-knowledge graph).

Replaces the v1 lossy extraction + cosine-similarity edge pipeline:
- extractor.py        keeps ALL packets (TCP/UDP/ICMP + control), records TCP flags + IP len
- flows.py            bidirectional 5-tuple flows + ~80 CICFlowMeter features incl flag counts + fan-out
- payload_features.py deterministic packet payload features (byte-hist + hashed n-gram + HTTP structural)
- signatures.py       curated OWASP-CRS-derived rules + flow signatures -> MITRE technique mapping
- evidence_edges.py   apply signatures -> evidence-grounded flow/packet -> technique edges
- graph_builder.py    assemble NPZ + meta.json artifact (artifact_version='v2')
- cli.py              single-command orchestrator (raw pcaps -> graph artifact)

Design doc:  docs/superpowers/specs/2026-05-24-evidence-grounded-graph-ids-design.md
Plan:        docs/superpowers/plans/2026-05-24-evidence-grounded-graph-ids.md
"""
