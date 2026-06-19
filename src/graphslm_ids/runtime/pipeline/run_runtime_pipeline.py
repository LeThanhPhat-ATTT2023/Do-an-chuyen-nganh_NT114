from __future__ import annotations

import argparse
from pathlib import Path

from graphslm_ids.runtime import FastPathPipeline, PipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay packets through the online IDS runtime.")
    parser.add_argument("--config", default="configs/pipeline.example.yaml")
    parser.add_argument("--input", required=True, help="PCAP/PCAPNG file to replay.")
    parser.add_argument("--no-worker", action="store_true", help="Do not start the slow-path worker.")
    parser.add_argument("--max-packets", type=int, default=None)
    return parser.parse_args()


def iter_packets(path: str | Path):
    """Yield per-packet dicts decoded with dpkt — the SAME decoder the offline
    extractor uses. Scapy's PcapReader mis-handles these captures' link layer
    ("unknown LL type") and falls back to Raw, which strips IP/TCP/payload and
    collapses everything into one bogus flow. dpkt + _decode_ip handles the DLT
    correctly, so the runtime sees the same packets the model trained on.
    """
    import socket

    import dpkt

    from graphslm_ids.offline.preprocessing.extractor import _decode_ip

    with open(path, "rb") as handle:
        try:
            reader = dpkt.pcap.Reader(handle)
            dlt = reader.datalink()
        except (ValueError, dpkt.dpkt.NeedData):
            return
        for ts, buf in reader:
            ip = _decode_ip(buf, dlt)
            if not isinstance(ip, dpkt.ip.IP):
                continue
            transport = ip.data
            src_port = 0
            dst_port = 0
            flags = 0
            if isinstance(transport, dpkt.tcp.TCP):
                protocol = "TCP"
                src_port = int(transport.sport)
                dst_port = int(transport.dport)
                flags = int(transport.flags)
                payload = bytes(transport.data)
            elif isinstance(transport, dpkt.udp.UDP):
                protocol = "UDP"
                src_port = int(transport.sport)
                dst_port = int(transport.dport)
                payload = bytes(transport.data)
            elif isinstance(transport, dpkt.icmp.ICMP):
                protocol = "ICMP"
                payload = bytes(transport.data)
            else:
                protocol = "OTHER"
                payload = b""
            try:
                src_ip = socket.inet_ntoa(ip.src)
                dst_ip = socket.inet_ntoa(ip.dst)
            except (OSError, ValueError):
                continue
            yield {
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": protocol,
                "timestamp": float(ts),
                "ip_len": int(getattr(ip, "len", 0) or 0),
                "tcp_flags": flags,
                "payload": payload,
            }


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig.from_yaml(args.config)
    pipeline = FastPathPipeline(cfg)
    worker = None
    if not args.no_worker:
        worker = pipeline.start_slow_worker()

    count = 0
    alerts = 0
    try:
        for packet in iter_packets(args.input):
            if args.max_packets is not None and count >= args.max_packets:
                break
            result = pipeline.on_packet(packet)
            count += 1
            if result.alert_id is not None:
                alerts += 1
                print(
                    f"[ALERT] {result.alert_id} flow={result.flow_id} "
                    f"label={result.label} confidence={result.confidence:.4f}"
                )
    finally:
        if worker is not None:
            pipeline.slow_queue.put(None)
            worker.join(timeout=60)

    print(f"[OK] Processed packets={count} alerts={alerts}")
    if cfg.graph_store.enabled:
        print(f"[OK] Graph store: {cfg.graph_store.root}")
    else:
        print(f"[OK] Cold store: {cfg.cold_store_path}")


if __name__ == "__main__":
    main()
