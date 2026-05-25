"""Build a tiny synthetic pcap for v2 unit tests.

The pcap intentionally contains a mix of packet kinds that the v2 pipeline must
handle correctly (and that the v1 lossy extractor would silently drop):
  - TCP SYN (no L4 payload)
  - TCP SYN+ACK (no L4 payload)
  - TCP PSH+ACK with HTTP-like payload (SQLi token, for signature tests)
  - UDP DNS-like packet
  - ICMP echo request
"""
from __future__ import annotations

import socket
from pathlib import Path

import dpkt


def _ip(src: str, dst: str, payload, proto: int) -> dpkt.ip.IP:
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst), p=proto)
    ip.data = payload
    ip.len = 20 + len(bytes(payload))
    return ip


def _eth(ip: dpkt.ip.IP) -> bytes:
    eth = dpkt.ethernet.Ethernet(
        src=b"\x00" * 6, dst=b"\xff" * 6, type=dpkt.ethernet.ETH_TYPE_IP
    )
    eth.data = ip
    return bytes(eth)


def build_demo_pcap(path: Path) -> None:
    """Write a 5-packet pcap covering TCP control, TCP with payload, UDP and ICMP."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        w = dpkt.pcap.Writer(f)
        t0 = 1_700_000_000.0
        # 1. TCP SYN, no payload
        tcp = dpkt.tcp.TCP(sport=44444, dport=80, flags=dpkt.tcp.TH_SYN)
        w.writepkt(_eth(_ip("10.0.0.1", "10.0.0.2", tcp, dpkt.ip.IP_PROTO_TCP)), ts=t0)
        # 2. TCP SYN+ACK reply, no payload (reverse direction)
        tcp = dpkt.tcp.TCP(
            sport=80, dport=44444, flags=dpkt.tcp.TH_SYN | dpkt.tcp.TH_ACK
        )
        w.writepkt(
            _eth(_ip("10.0.0.2", "10.0.0.1", tcp, dpkt.ip.IP_PROTO_TCP)), ts=t0 + 0.001
        )
        # 3. TCP PSH+ACK with HTTP+SQLi payload (forward dir)
        tcp = dpkt.tcp.TCP(
            sport=44444,
            dport=80,
            flags=dpkt.tcp.TH_PUSH | dpkt.tcp.TH_ACK,
            data=b"GET /?q=' OR 1=1-- HTTP/1.1\r\n\r\n",
        )
        w.writepkt(
            _eth(_ip("10.0.0.1", "10.0.0.2", tcp, dpkt.ip.IP_PROTO_TCP)), ts=t0 + 0.002
        )
        # 4. UDP DNS-like
        udp = dpkt.udp.UDP(sport=33333, dport=53, data=b"\x00" * 30)
        udp.ulen = 8 + 30
        w.writepkt(
            _eth(_ip("10.0.0.1", "8.8.8.8", udp, dpkt.ip.IP_PROTO_UDP)), ts=t0 + 0.003
        )
        # 5. ICMP echo request
        icmp = dpkt.icmp.ICMP(
            type=8, data=dpkt.icmp.ICMP.Echo(id=1, seq=1, data=b"ping")
        )
        w.writepkt(
            _eth(_ip("10.0.0.1", "10.0.0.3", icmp, dpkt.ip.IP_PROTO_ICMP)),
            ts=t0 + 0.004,
        )
