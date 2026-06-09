"""DECISIVE test #3 (proper model): the TRUE side-channel ceiling for the
encrypted web cluster, measured with a Deep-Fingerprinting-style 1D-CNN on full
directional packet sequences -- NOT a weak RandomForest on 32 packets.

Per flow we extract up to L=256 packets, each as a 3-channel step:
  [ signed_size_norm, log1p_iat, signed_tls_record_size_norm ]
Direction sign: +client->server, -server->client. This is the DF/ET-BERT family
of features that fingerprints websites through Tor at ~98%.

Outputs per-class F1 for {CommandInjection, XSS, Uploading_Attack, SqlInjection}
and the 3-class encrypted-cluster macro-F1. This is the honest ceiling of what a
deployment-realistic (no decryption, no campaign-leakage) model can do.
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import dpkt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

RAW = Path("/home/ubuntu/dataset/raw")
FILES = {
    "CommandInjection": RAW / "CommandInjection" / "CommandInjection.pcap",
    "XSS": RAW / "XSS" / "XSS.pcap",
    "Uploading_Attack": RAW / "Uploading_Attack" / "Uploading_Attack.pcap",
    "SqlInjection": RAW / "SqlInjection" / "SqlInjection.pcap",
}
L = 256
MIN_FLOW_PKTS = 6
MAX_PKTS = 1_200_000
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse(pcap_path: Path):
    flows = collections.defaultdict(list)
    first_dir = {}
    with open(pcap_path, "rb") as fh:
        try:
            pcap = dpkt.pcap.Reader(fh)
        except ValueError:
            fh.seek(0); pcap = dpkt.pcapng.Reader(fh)
        n = 0
        for ts, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf); ip = eth.data; l4 = ip.data
            except Exception:
                continue
            if not isinstance(ip, dpkt.ip.IP) or not isinstance(l4, dpkt.tcp.TCP):
                continue
            a, b = (ip.src, l4.sport), (ip.dst, l4.dport)
            key = (a, b) if a < b else (b, a)
            if key not in first_dir:
                first_dir[key] = (ip.src, ip.dst)
            sd = first_dir[key]
            sign = 1.0 if (ip.src, ip.dst) == sd else -1.0
            data = bytes(l4.data); off = 0; rec = 0
            while off + 5 <= len(data):
                if data[off] not in (0x14, 0x15, 0x16, 0x17):
                    break
                rl = (data[off + 3] << 8) | data[off + 4]
                rec = rl; off += 5 + rl
                if rl == 0:
                    break
            flows[key].append((ts, sign * float(ip.len), sign * float(rec)))
            n += 1
            if n >= MAX_PKTS:
                break
    return flows


def seq_tensor(events):
    events = sorted(events, key=lambda e: e[0])[:L]
    ts = np.array([e[0] for e in events])
    size = np.array([e[1] for e in events], dtype=np.float32)
    rec = np.array([e[2] for e in events], dtype=np.float32)
    iat = np.zeros_like(size)
    if len(ts) > 1:
        d = np.diff(ts); iat[1:] = np.sign(size[1:]) * np.log1p(np.abs(d))
    size = np.sign(size) * np.log1p(np.abs(size)) / 10.0
    rec = np.sign(rec) * np.log1p(np.abs(rec)) / 10.0
    ch = np.stack([size, rec, iat], axis=0)  # (3, len)
    out = np.zeros((3, L), dtype=np.float32)
    out[:, : ch.shape[1]] = ch
    return out


class DFNet(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(
                nn.Conv1d(ci, co, 7, padding=3), nn.BatchNorm1d(co), nn.ReLU(),
                nn.Conv1d(co, co, 7, padding=3), nn.BatchNorm1d(co), nn.ReLU(),
                nn.MaxPool1d(3, 2), nn.Dropout(0.2))
        self.net = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.4),
                                  nn.Linear(128, n_cls))

    def forward(self, x):
        return self.head(self.net(x))


def main():
    rows = []
    for cls, p in FILES.items():
        if not p.exists():
            print("skip", cls); continue
        fl = parse(p); kept = 0
        for k, ev in fl.items():
            if len(ev) < MIN_FLOW_PKTS:
                continue
            rows.append((cls, seq_tensor(ev))); kept += 1
        print(f"[{cls:18s}] flows={len(fl):6d} kept={kept}", flush=True)

    labels = sorted({r[0] for r in rows}); lab2i = {c: i for i, c in enumerate(labels)}
    X = np.stack([r[1] for r in rows]); y = np.array([lab2i[r[0]] for r in rows])
    print("total", X.shape, "per-class", {c: int((y == lab2i[c]).sum()) for c in labels}, flush=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    cls_count = np.bincount(ytr, minlength=len(labels))
    w = torch.tensor((cls_count.sum() / (len(labels) * np.maximum(cls_count, 1))),
                     dtype=torch.float32, device=DEV)
    model = DFNet(len(labels)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xtr_t = torch.tensor(Xtr, device=DEV); ytr_t = torch.tensor(ytr, device=DEV)
    Xte_t = torch.tensor(Xte, device=DEV)
    n = len(Xtr_t); bs = 256
    for ep in range(60):
        model.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                pr = model(Xte_t).argmax(1).cpu().numpy()
            mf = f1_score(yte, pr, average="macro")
            print(f"  epoch {ep+1:02d} loss={loss.item():.3f} web macro-F1={mf:.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        pr = model(Xte_t).argmax(1).cpu().numpy()
    mf = f1_score(yte, pr, average="macro"); perc = f1_score(yte, pr, average=None)
    print("\n=== DF-CNN side-channel ceiling (honest, no decryption) ===")
    print(f"  web macro-F1 = {mf:.4f}  (RF baseline 0.606, byte-4gram 0.591)")
    print("  per-class F1:", {labels[i]: round(float(perc[i]), 3) for i in range(len(labels))})
    print("  confusion order", labels); print(confusion_matrix(yte, pr))
    enc = [lab2i[c] for c in labels if c != "SqlInjection"]
    m = np.isin(yte, enc)
    print(f"  [encrypted cluster CmdInj/XSS/Upload] macro-F1 = "
          f"{f1_score(yte[m], pr[m], average='macro', labels=enc):.4f}")


if __name__ == "__main__":
    main()
