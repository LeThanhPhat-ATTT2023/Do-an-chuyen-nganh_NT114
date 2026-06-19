# System Execution Flows

This file contains two vertical Mermaid diagrams for thesis/report usage.
They describe the **v3 (`v3_ob`) pipeline**: zero learned encoders besides HGT
itself, MSEE evidence edges (PMI + procedure matcher + flow consensus), and an
evidence-grounded + verified SLM report on the slow path.

## 1) Training Pipeline (Offline, local CPU)

```mermaid
flowchart TD
    A[Raw labeled PCAP dataset, 18 classes] --> B[Extractor: ALL packets plus TCP flags, IP len, direction]
    B --> C[Flows: bidirectional 5-tuple plus 80 CICFlowMeter features]
    C --> D[Split: temporal AND random stratified, seed 42]

    B --> E[Tokenizer: deterministic byte n-gram plus HTTP text tokens]
    E --> F[PMI learner: candidate generation plus L1 multinomial LR]
    B --> G[Procedure matcher: Aho-Corasick over MITRE STIX literals]
    C --> H[Flow consensus: behavioral signature boost]

    F --> I[Ensemble MSEE: PMI plus procedure plus flow consensus -> edge weights]
    G --> I
    H --> I

    I --> J[Graph builder: flow, packet, host, technique, tactic nodes plus typed evidence edges plus hierarchy edges]
    D --> J
    J --> K[Artifact v3_ob: graph.npz plus splits plus pmi_table]

    K --> L[Train HGT classifier plus GCL aux loss plus EACS self-relabeling]
    L --> M[Calibrate thresholds on clean VAL, apply to TEST]
    M --> N[Eval BOTH random and temporal splits -> report the GAP]
    N --> O[Export runtime artifacts: HGT checkpoint, pmi_table, mitre index, thresholds]
```

Notes:
- **No learned encoders besides HGT.** Packet→technique edges come from the
  Multi-Source Evidence Ensemble (MSEE): PMI counting + convex L1-LR refinement +
  Aho-Corasick procedure matching. The legacy SecureBERT-cosine `matches_technique`
  edge of v1 is removed.
- Stages are deterministic given seed 42; only HGT trains.
- EACS self-relabels suspect web-attack flows lacking MITRE evidence; the clean
  answer key and EACS anchor mask are eval-only and never enter any loss.

## 2) Runtime Pipeline (Online, fast path + slow path)

```mermaid
flowchart TD
    A[Live network traffic] --> B[Stream parser plus flow tracker]
    B --> C[Online payload extractor plus flow stats]
    C --> D[Runtime edge assigner: PMI table plus procedure matcher with provenance]
    D --> E[Hot graph buffer: packets, flows, MITRE top-k plus provenance]
    E --> F[Subgraph builder: current heterogeneous window]

    F --> G[Fast path: HGT inference]
    G --> H{Policy engine}
    H -->|Low risk| I[Allow]
    H -->|Medium risk| J[Alert or rate limit]
    H -->|High risk| K[Block or drop via firewall]

    G --> L[Slow path queue, on alert]
    L --> M[Hot buffer adapter plus evidence builder -> v3 evidence bundle]
    M --> N[Graph serializer -> GRAPH-TEXT with family, weight, src, tokens, evidence_id]
    N --> O[SLM explainer reads graph-text]
    O --> P[Graph verifier VG2R: check each claim against evidence_id]
    P -->|grounded| Q[Grounded XAI report]
    P -->|ungrounded| R[Repair or fallback template]
    Q --> S[SOC dashboard]
    R --> S
    S --> T[Analyst feedback]
    T --> U[Periodic retraining trigger]
```

Notes:
- The fast-path runtime edge assigner reuses the **same** PMI table + procedure
  matcher as the offline build, carrying token/literal **provenance** through the
  hot buffer so the SLM report is auditable.
- The SLM does **not** read the HGT label alone. It reads the serialized evidence
  subgraph (**graph-text**) which contains the HGT decision *plus* the supporting
  evidence edges (`family=... w=... src=pmi tokens=...`) and per-element
  `evidence_id`s.
- **VG2R verifier** grounds every claim in the report against the graph-text
  `evidence_id`s and scores `numeric_accuracy`; ungrounded claims trigger repair
  or a deterministic fallback template instead of being emitted.
