"""DECISIVE test #2: can ENCRYPTED-TRAFFIC side-channel features separate the
confused web classes (CommandInjection / XSS / Uploading_Attack) where payload
content is TLS-encrypted and byte-4gram caps at 0.59?

Side-channel features (no decryption needed):
  * signed packet-length sequence (first N pkts, +fwd / -bwd)
  * TLS record-length sequence (cleartext 5-byte TLS record headers)
  * directional burst + volume aggregates
  * basic timing

If web macro-F1 jumps 0.59 -> >0.80 => encrypted-traffic features are the cure.
If Upload alone jumps but CmdInj/XSS stay confused => partial ceiling.
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import dpkt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

RAW = Path("/home/ubuntu/dataset/raw")
FILES = {
    "CommandInjection": RAW / "CommandInjection" / "CommandInjection.pcap",
    "XSS": RAW / "XSS" / "XSS.pcap",
    "Uploading_Attack": RAW / "Uploading_Attack" / "Uploading_Attack.pcap",
    "SqlInjection": RAW / "SqlInjection" / "SqlInjection.pcap",
}

SEQ = 32             # packets per flow in the length sequence
MAX_PKTS = 400000    # cap packets read per pcap
MIN_FLOW_PKTS = 4    # ignore tiny flows


def parse_flows(pcap_path: Path):
    """Return dict flow_key -> list of (ts, signed_iplen, tls_rec_sizes)."""
    flows: dict = collections.defaultdict(list)
    first_dir: dict = {}
    with open(pcap_path, "rb") as fh:
        try:
            pcap = dpkt.pcap.Reader(fh)
        except ValueError:
            fh.seek(0)
            pcap = dpkt.pcapng.Reader(fh)
        n = 0
        for ts, buf in pcap:
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
            # canonical bidirectional 5-tuple key
            a = (ip.src, l4.sport)
            b = (ip.dst, l4.dport)
            key = (a, b) if a < b else (b, a)
            if key not in first_dir:
                first_dir[key] = (ip.src, ip.dst)
            sd = first_dir[key]
            sign = 1 if (ip.src, ip.dst) == sd else -1
            # TLS record sizes from cleartext record headers
            rec_sizes = []
            data = bytes(l4.data)
            off = 0
            while off + 5 <= len(data):
                ctype = data[off]
                if ctype not in (0x14, 0x15, 0x16, 0x17):
                    break
                rlen = (data[off + 3] << 8) | data[off + 4]
                rec_sizes.append(rlen)
                off += 5 + rlen
                if rlen == 0:
                    break
            flows[key].append((ts, sign * int(ip.len), rec_sizes))
            n += 1
            if n >= MAX_PKTS:
                break
    return flows


def flow_features(events) -> np.ndarray:
    events = sorted(events, key=lambda e: e[0])
    sizes = [e[1] for e in events]
    ts = [e[0] for e in events]
    # signed length sequence (pad/truncate)
    seq = sizes[:SEQ] + [0] * max(0, SEQ - len(sizes))
    seq = np.array(seq[:SEQ], dtype=np.float32)
    fwd = np.array([s for s in sizes if s > 0], dtype=np.float32)
    bwd = np.array([-s for s in sizes if s < 0], dtype=np.float32)
    # TLS record-size aggregates
    recs = [r for e in events for r in e[2]]
    recs = np.array(recs, dtype=np.float32) if recs else np.array([0.0], dtype=np.float32)
    iat = np.diff(ts) if len(ts) > 1 else np.array([0.0])
    agg = np.array([
        len(sizes),
        fwd.sum(), bwd.sum(),
        fwd.sum() / (bwd.sum() + 1.0),          # up/down ratio
        fwd.mean() if fwd.size else 0.0, fwd.std() if fwd.size else 0.0,
        bwd.mean() if bwd.size else 0.0, bwd.std() if bwd.size else 0.0,
        fwd.max() if fwd.size else 0.0, bwd.max() if bwd.size else 0.0,
        len(fwd), len(bwd),
        recs.mean(), recs.std(), recs.max(), len(recs),
        float(np.mean(iat)), float(np.std(iat)), float(np.max(iat)),
    ], dtype=np.float32)
    return np.concatenate([seq, agg])


def main() -> None:
    rows = []
    for cls, path in FILES.items():
        if not path.exists():
            print(f"[skip] {cls}: missing"); continue
        flows = parse_flows(path)
        kept = 0
        for key, ev in flows.items():
            if len(ev) < MIN_FLOW_PKTS:
                continue
            rows.append((cls, flow_features(ev)))
            kept += 1
        print(f"[{cls:18s}] flows={len(flows):6d} kept(>= {MIN_FLOW_PKTS}pkts)={kept}")

    labels = sorted({r[0] for r in rows})
    lab2i = {c: i for i, c in enumerate(labels)}
    X = np.array([r[1] for r in rows]); y = np.array([lab2i[r[0]] for r in rows])
    # balance: cap per class for fair LR/RF
    print(f"\ntotal flows={len(y)} dim={X.shape[1]} per-class="
          f"{ {c:int((y==lab2i[c]).sum()) for c in labels} }")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced",
        n_jobs=-1, random_state=42)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    mf = f1_score(yte, pred, average="macro")
    perc = f1_score(yte, pred, average=None)
    print("\n=== DECISIVE: web separability from ENCRYPTED side-channel ===")
    print(f"  web macro-F1 = {mf:.4f}  (byte-4gram baseline = 0.591)")
    print("  per-class F1:",
          {labels[i]: round(float(perc[i]), 3) for i in range(len(labels))})
    print("  confusion (rows=true,cols=pred), order:", labels)
    print(confusion_matrix(yte, pred))
    # 3-class encrypted cluster only (drop SQLi which has plaintext)
    enc = [c for c in labels if c != "SqlInjection"]
    m3 = np.isin(yte, [lab2i[c] for c in enc])
    if m3.any():
        mf3 = f1_score(yte[m3], pred[m3], average="macro",
                       labels=[lab2i[c] for c in enc])
        print(f"\n  [encrypted-only CmdInj/XSS/Upload] macro-F1 = {mf3:.4f}")


if __name__ == "__main__":
    main()
