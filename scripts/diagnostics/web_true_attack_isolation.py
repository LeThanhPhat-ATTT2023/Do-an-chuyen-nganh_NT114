"""DECISIVE test #5: isolate the TRUE attack flows from background IoT-cloud
noise. The TLS-handshake test revealed CmdInj/XSS/Upload pcaps are dominated by
benign IoT cloud traffic (smartthings/netatmo/alexa) -- identical across classes
-> that shared noise is the confusion floor. The actual web attack targets a
LOCAL web app over plaintext HTTP.

This script:
  1. Collects plaintext HTTP requests (GET/POST/...) per class.
  2. Shows dst IP/port + Host header + request-line samples (reveal the target).
  3. Measures separability on TRUE attack requests (HTTP content only).
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

import numpy as np
import dpkt
import socket
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

RAW = Path("/home/ubuntu/dataset/raw")
FILES = {
    "CommandInjection": RAW / "CommandInjection" / "CommandInjection.pcap",
    "XSS": RAW / "XSS" / "XSS.pcap",
    "Uploading_Attack": RAW / "Uploading_Attack" / "Uploading_Attack.pcap",
    "SqlInjection": RAW / "SqlInjection" / "SqlInjection.pcap",
}
HTTP_RE = re.compile(rb"^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH) ")
MAX_PKTS = 2_000_000


def ipstr(b):
    try:
        return socket.inet_ntoa(b)
    except Exception:
        return "?"


def is_private(ip: str) -> bool:
    return (ip.startswith("10.") or ip.startswith("192.168.")
            or any(ip.startswith(f"172.{i}.") for i in range(16, 32)))


def collect(pcap_path: Path):
    reqs = []        # (dst_ip, dport, host, request_line, full_lowered_payload)
    dst_counter = collections.Counter()
    with open(pcap_path, "rb") as fh:
        try:
            pcap = dpkt.pcap.Reader(fh)
        except ValueError:
            fh.seek(0); pcap = dpkt.pcapng.Reader(fh)
        n = 0
        for ts, buf in pcap:
            n += 1
            if n >= MAX_PKTS:
                break
            try:
                eth = dpkt.ethernet.Ethernet(buf); ip = eth.data; l4 = ip.data
            except Exception:
                continue
            if not isinstance(ip, dpkt.ip.IP) or not isinstance(l4, dpkt.tcp.TCP):
                continue
            d = bytes(l4.data)
            if not d or not HTTP_RE.match(d):
                continue
            dip = ipstr(ip.dst)
            host = ""
            m = re.search(rb"[Hh]ost:\s*([^\r\n]+)", d)
            if m:
                host = m.group(1).decode("latin1", "ignore").strip()
            line = d.split(b"\r\n", 1)[0].decode("latin1", "ignore")[:120]
            reqs.append((dip, l4.dport, host, line, d.lower()))
            dst_counter[(dip, l4.dport)] += 1
    return reqs, dst_counter


def main():
    all_reqs = {}
    for cls, p in FILES.items():
        if not p.exists():
            print("skip", cls); continue
        reqs, dstc = collect(p)
        all_reqs[cls] = reqs
        priv = sum(c for (ip, pt), c in dstc.items() if is_private(ip))
        print(f"\n[{cls}] plaintext-HTTP requests={len(reqs)} "
              f"(to private/local dst={priv})")
        print("   top dst (ip,port):", dstc.most_common(4))
        # show a few request lines to LOCAL targets (the real attack)
        shown = 0
        for dip, dport, host, line, _ in reqs:
            if is_private(dip):
                print(f"     -> {dip}:{dport} Host={host[:25]} | {line}")
                shown += 1
                if shown >= 4:
                    break

    # Separability on TRUE attack requests: keep only requests to PRIVATE dst
    rows = []
    for cls, reqs in all_reqs.items():
        for dip, dport, host, line, payload in reqs:
            if is_private(dip):
                rows.append((cls, payload))
    print("\n=== TRUE-attack (local-target plaintext HTTP) separability ===")
    if len(rows) < 40:
        print(f"  only {len(rows)} local-target HTTP requests -> attack likely "
              f"NOT in plaintext-to-local; see dst breakdown above.")
        return
    labels = sorted({r[0] for r in rows}); lab2i = {c: i for i, c in enumerate(labels)}
    print("  per-class local-HTTP counts:",
          {c: sum(1 for r in rows if r[0] == c) for c in labels})
    texts = [r[1].decode("latin1", "ignore") for r in rows]
    y = np.array([lab2i[r[0]] for r in rows])
    X = HashingVectorizer(n_features=4096, analyzer="char_wb",
                          ngram_range=(3, 5)).transform(texts)
    if min(np.bincount(y)) < 5:
        print("  some class <5 local requests; counts:",
              {labels[i]: int((y == i).sum()) for i in range(len(labels))})
        return
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
    pr = lr.predict(Xte)
    mf = f1_score(yte, pr, average="macro"); perc = f1_score(yte, pr, average=None)
    print(f"  web macro-F1 (TRUE attacks) = {mf:.4f}")
    print("  per-class F1:", {labels[i]: round(float(perc[i]), 3) for i in range(len(labels))})
    print("  confusion", labels); print(confusion_matrix(yte, pr))


if __name__ == "__main__":
    main()
