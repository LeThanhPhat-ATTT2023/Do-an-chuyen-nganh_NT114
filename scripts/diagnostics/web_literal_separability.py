"""DECISIVE test: do exact-match semantic literals separate the confused web
classes (CommandInjection / XSS / Uploading_Attack), where length-normalized
byte-4gram features cap at ~0.59?

Reads raw TCP payloads from the 4 web-attack PCAPs, builds a per-packet
exact-match literal-presence vector, aggregates to flow (max-pool), and runs a
plain LogisticRegression to measure the achievable web-class macro-F1 from
exact-match content alone.

If macro-F1 jumps 0.59 -> >0.80 => semantic signatures are the cure, 0.9 is in
reach. If it stays ~0.6 => the discriminating content is not in the captured
payload and no model can separate these classes.
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

import numpy as np
import dpkt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

RAW = Path("data/raw")
FILES = {
    "CommandInjection": RAW / "CommandInjection" / "CommandInjection.pcap",
    "XSS": RAW / "XSS" / "XSS.pcap",
    "Uploading_Attack": RAW / "Uploading_Attack" / "Uploading_Attack.pcap",
    "SqlInjection": RAW / "SqlInjection" / "SqlInjection.pcap",
}

# Per-family discriminating literals (lowercased, raw + url-encoded variants).
LIT = {
    "xss": [b"<script", b"</script", b"onerror", b"onload=", b"javascript:",
            b"alert(", b"<svg", b"<img", b"<iframe", b"document.cookie",
            b"%3cscript", b"onmouseover"],
    "sqli": [b"union select", b"union%20select", b"or 1=1", b"or%201=1",
             b"' or '", b"'or'", b"information_schema", b"sleep(", b"concat(",
             b"select * from", b" from ", b"--", b"/*", b"' and "],
    "cmd": [b";cat", b"|cat", b";ls", b"; ls", b"$(", b"&&", b"/etc/passwd",
            b"whoami", b"wget ", b"curl ", b"|bash", b"|sh", b"`", b"%3b",
            b"; sleep", b"ping -c", b"uname"],
    "upload": [b"content-disposition", b"filename=", b"multipart/form-data",
               b"boundary=", b".php", b".jsp", b".asp", b".exe",
               b"application/octet-stream", b"content-type: image"],
}
FAM_ORDER = ["xss", "sqli", "cmd", "upload"]
ALL_LITS = [(fam, lit) for fam in FAM_ORDER for lit in LIT[fam]]

MAX_PAYLOAD_PKTS = 6000  # per class, payload-bearing only


def iter_payloads(pcap_path: Path):
    """Yield (flow_key, lowercased_tcp_payload) for payload-bearing TCP pkts."""
    with open(pcap_path, "rb") as fh:
        try:
            pcap = dpkt.pcap.Reader(fh)
        except ValueError:
            fh.seek(0)
            pcap = dpkt.pcapng.Reader(fh)
        for _ts, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue
            ip = eth.data
            if not isinstance(ip, dpkt.ip.IP):
                continue
            l4 = ip.data
            if not isinstance(l4, dpkt.tcp.TCP):
                continue
            payload = bytes(l4.data)
            if not payload:
                continue
            key = (ip.src, ip.dst, l4.sport, l4.dport)
            yield key, payload.lower()


def lit_vector(payload: bytes) -> np.ndarray:
    v = np.zeros(len(ALL_LITS), dtype=np.float32)
    for i, (_fam, lit) in enumerate(ALL_LITS):
        if lit in payload:
            v[i] = 1.0
    return v


def main() -> None:
    # Aggregate to flow level (max over packets), keep per-class flows.
    rows = []  # (label, flowvec)
    coverage = collections.defaultdict(lambda: [0, 0])  # cls -> [hit_pkts, tot_pkts]
    for cls, path in FILES.items():
        if not path.exists():
            print(f"[skip] {cls}: {path} missing")
            continue
        flows: dict = {}
        n = 0
        for key, payload in iter_payloads(path):
            v = lit_vector(payload)
            coverage[cls][1] += 1
            if v.any():
                coverage[cls][0] += 1
            if key in flows:
                np.maximum(flows[key], v, out=flows[key])
            else:
                flows[key] = v.copy()
            n += 1
            if n >= MAX_PAYLOAD_PKTS:
                break
        for v in flows.values():
            rows.append((cls, v))
        print(f"[{cls:18s}] payload pkts={n:6d} flows={len(flows):5d} "
              f"pkt-literal-coverage={coverage[cls][0]/max(coverage[cls][1],1):.1%}")

    labels = sorted({r[0] for r in rows})
    lab2i = {c: i for i, c in enumerate(labels)}
    X = np.array([r[1] for r in rows])
    y = np.array([lab2i[r[0]] for r in rows])
    print(f"\nflows total={len(y)} feat-dim={X.shape[1]} classes={labels}")
    print("per-class flow counts:",
          {c: int((y == lab2i[c]).sum()) for c in labels})

    # Which literals fire per class (mean presence) -> discrimination evidence
    print("\n=== mean literal-presence per class (top discriminators) ===")
    for c in labels:
        sub = X[y == lab2i[c]]
        means = sub.mean(0)
        top = np.argsort(means)[::-1][:6]
        desc = [(ALL_LITS[j][1].decode("latin1"), round(float(means[j]), 2))
                for j in top if means[j] > 0]
        print(f"  {c:18s}: {desc}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=3000, class_weight="balanced")
    lr.fit(sc.transform(Xtr), ytr)
    pred = lr.predict(sc.transform(Xte))
    mf = f1_score(yte, pred, average="macro")
    perc = f1_score(yte, pred, average=None)
    print("\n=== DECISIVE RESULT: exact-literal web separability ===")
    print(f"  web macro-F1 = {mf:.4f}")
    print("  per-class F1:",
          {labels[i]: round(float(perc[i]), 3) for i in range(len(labels))})
    print("  confusion matrix (rows=true, cols=pred), order:", labels)
    print(confusion_matrix(yte, pred))


if __name__ == "__main__":
    main()
