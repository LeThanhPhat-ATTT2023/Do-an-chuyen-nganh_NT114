"""DECISIVE test #4: TLS HANDSHAKE fingerprint separability for the encrypted
web cluster. ClientHello is cleartext even over HTTPS -- if the attack types use
different client tools, JA3 / cipher-suites / extensions / SNI / ALPN will
separate them HONESTLY (tool fingerprint generalizes; not campaign leakage).

Extracts from each flow's ClientHello: TLS version, cipher-suite list, extension
list, elliptic curves, EC point formats (-> JA3 string + md5), SNI, ALPN.
Reports per-class JA3/SNI value distributions (the key diagnostic: do classes
differ at all?) and LR/RF separability on handshake features.
"""
from __future__ import annotations

import collections
import hashlib
from pathlib import Path

import numpy as np
import dpkt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

RAW = Path("/home/ubuntu/dataset/raw")
FILES = {
    "CommandInjection": RAW / "CommandInjection" / "CommandInjection.pcap",
    "XSS": RAW / "XSS" / "XSS.pcap",
    "Uploading_Attack": RAW / "Uploading_Attack" / "Uploading_Attack.pcap",
    "SqlInjection": RAW / "SqlInjection" / "SqlInjection.pcap",
}
GREASE = {0x0a0a, 0x1a1a, 0x2a2a, 0x3a3a, 0x4a4a, 0x5a5a, 0x6a6a, 0x7a7a,
          0x8a8a, 0x9a9a, 0xaaaa, 0xbaba, 0xcaca, 0xdada, 0xeaea, 0xfafa}
MAX_PKTS = 1_500_000


def ja3_from_hello(ch) -> tuple[str, str, str]:
    """Return (ja3_str, sni, alpn) from a dpkt TLSClientHello."""
    ver = ch.version
    ciphers = []
    for c in getattr(ch, "ciphersuites", []):
        code = getattr(c, "code", c)
        if isinstance(code, int) and code not in GREASE:
            ciphers.append(code)
    exts, curves, ecpf, sni, alpn = [], [], [], "", ""
    try:
        for ext_type, ext_data in ch.extensions:
            if ext_type in GREASE:
                continue
            exts.append(ext_type)
            if ext_type == 0x0000 and len(ext_data) > 5:  # SNI
                try:
                    sni = ext_data[5:].decode("latin1", "ignore")
                except Exception:
                    sni = ""
            elif ext_type == 0x000a:  # supported groups / curves
                if len(ext_data) >= 2:
                    n = int.from_bytes(ext_data[:2], "big")
                    curves = [int.from_bytes(ext_data[2+i:4+i], "big")
                              for i in range(0, n, 2)]
            elif ext_type == 0x000b:  # ec point formats
                if ext_data:
                    ln = ext_data[0]
                    ecpf = list(ext_data[1:1+ln])
            elif ext_type == 0x0010 and len(ext_data) > 2:  # ALPN
                alpn = ext_data[3:].decode("latin1", "ignore")
    except Exception:
        pass
    ja3 = "{},{},{},{},{}".format(
        ver,
        "-".join(map(str, ciphers)),
        "-".join(map(str, exts)),
        "-".join(map(str, curves)),
        "-".join(map(str, ecpf)),
    )
    return ja3, sni, alpn


def extract(pcap_path: Path):
    """Yield one (ja3, ja3hash, sni, alpn, ncipher, next) per ClientHello flow."""
    out = []
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
            if not isinstance(l4, dpkt.tcp.TCP):
                continue
            data = bytes(l4.data)
            if len(data) < 6 or data[0] != 0x16:  # handshake record
                continue
            try:
                rec = dpkt.ssl.TLSRecord(data)
                hs = dpkt.ssl.TLSHandshake(rec.data)
                if not isinstance(hs.data, dpkt.ssl.TLSClientHello):
                    continue
                ja3, sni, alpn = ja3_from_hello(hs.data)
                out.append((ja3, hashlib.md5(ja3.encode()).hexdigest(), sni, alpn))
            except Exception:
                continue
    return out


def main():
    per_class = {}
    for cls, p in FILES.items():
        if not p.exists():
            print("skip", cls); continue
        rows = extract(p)
        per_class[cls] = rows
        ja3s = collections.Counter(r[1] for r in rows)
        snis = collections.Counter(r[2] for r in rows)
        alpns = collections.Counter(r[3] for r in rows)
        print(f"\n[{cls}] ClientHellos={len(rows)}")
        print(f"   distinct JA3={len(ja3s)} | top: {ja3s.most_common(3)}")
        print(f"   distinct SNI={len(snis)} | top: {[ (s[:30], c) for s,c in snis.most_common(3)]}")
        print(f"   ALPN top: {alpns.most_common(3)}")

    # Separability: hash JA3 + SNI + ALPN into features, RF
    allrows = [(cls, r) for cls, rows in per_class.items() for r in rows]
    labels = sorted(per_class.keys()); lab2i = {c: i for i, c in enumerate(labels)}
    feats = [{"ja3=" + r[1]: 1.0, "sni=" + r[2]: 1.0, "alpn=" + r[3]: 1.0}
             for _, r in allrows]
    X = FeatureHasher(n_features=512).transform(feats).toarray().astype(np.float32)
    y = np.array([lab2i[c] for c, _ in allrows])
    print("\ntotal hellos", len(y), "per-class",
          {c: int((y == lab2i[c]).sum()) for c in labels})
    if len(set(y)) < 2 or min(np.bincount(y)) < 5:
        print("not enough per-class ClientHellos for separability test"); return
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                n_jobs=-1, random_state=42).fit(Xtr, ytr)
    pr = rf.predict(Xte)
    mf = f1_score(yte, pr, average="macro"); perc = f1_score(yte, pr, average=None)
    print("\n=== TLS-handshake-fingerprint web separability ===")
    print(f"  web macro-F1 = {mf:.4f}")
    print("  per-class F1:", {labels[i]: round(float(perc[i]), 3) for i in range(len(labels))})
    print("  confusion", labels); print(confusion_matrix(yte, pr))
    enc = [lab2i[c] for c in labels if c != "SqlInjection"]
    m = np.isin(yte, enc)
    if m.any():
        print(f"  [encrypted cluster] macro-F1 = "
              f"{f1_score(yte[m], pr[m], average='macro', labels=enc):.4f}")


if __name__ == "__main__":
    main()
