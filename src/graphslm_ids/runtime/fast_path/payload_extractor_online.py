from __future__ import annotations

import binascii
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ExtractedPayload:
    payload_u8: np.ndarray
    hex_64: str
    ascii_64: str
    raw_len: int
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    timestamp: float
    # v3 CICFlowMeter parity: IP total length + raw TCP flag bits, needed so the
    # online flow-feature builder reproduces the 79-dim offline vector.
    ip_len: int = 0
    tcp_flags: int = 0


class PayloadExtractor:
    """Extract fixed-length payload vectors from live Scapy packets or packet-like dicts."""

    def __init__(self, payload_length: int = 256, preview_bytes: int = 64) -> None:
        self.payload_length = int(payload_length)
        self.preview_bytes = int(preview_bytes)

    def extract(self, pkt: Any) -> ExtractedPayload:
        meta = _extract_metadata(pkt)
        raw_payload = _extract_payload_bytes(pkt)
        payload_u8 = _truncate_and_pad_payload(raw_payload, payload_length=self.payload_length)
        preview = raw_payload[: self.preview_bytes]
        return ExtractedPayload(
            payload_u8=payload_u8,
            hex_64=binascii.hexlify(preview).decode("ascii"),
            ascii_64=_bytes_to_ascii(preview),
            raw_len=len(raw_payload),
            src_ip=str(meta["src_ip"]),
            dst_ip=str(meta["dst_ip"]),
            src_port=int(meta["src_port"]),
            dst_port=int(meta["dst_port"]),
            protocol=str(meta["protocol"]).upper(),
            timestamp=float(meta["timestamp"]),
            ip_len=int(meta.get("ip_len", 0) or 0),
            tcp_flags=int(meta.get("tcp_flags", 0) or 0),
        )


def _extract_metadata(pkt: Any) -> dict[str, Any]:
    if isinstance(pkt, dict):
        return {
            "src_ip": pkt.get("src_ip", pkt.get("src", "unknown")),
            "dst_ip": pkt.get("dst_ip", pkt.get("dst", "unknown")),
            "src_port": pkt.get("src_port", pkt.get("sport", 0)),
            "dst_port": pkt.get("dst_port", pkt.get("dport", 0)),
            "protocol": pkt.get("protocol", pkt.get("proto", "OTHER")),
            "timestamp": pkt.get("timestamp", pkt.get("time", 0.0)),
            "ip_len": pkt.get("ip_len", 0),
            "tcp_flags": pkt.get("tcp_flags", pkt.get("flags", 0)),
        }

    try:
        from scapy.layers.inet import IP, TCP, UDP
    except ImportError:
        return {
            "src_ip": getattr(pkt, "src_ip", "unknown"),
            "dst_ip": getattr(pkt, "dst_ip", "unknown"),
            "src_port": getattr(pkt, "src_port", 0),
            "dst_port": getattr(pkt, "dst_port", 0),
            "protocol": getattr(pkt, "protocol", "OTHER"),
            "timestamp": getattr(pkt, "timestamp", getattr(pkt, "time", 0.0)),
        }

    src_ip = "unknown"
    dst_ip = "unknown"
    protocol = "OTHER"
    src_port = 0
    dst_port = 0
    ip_len = 0
    tcp_flags = 0
    if IP in pkt:
        ip_layer = pkt[IP]
        src_ip = str(ip_layer.src)
        dst_ip = str(ip_layer.dst)
        ip_len = int(getattr(ip_layer, "len", 0) or 0)
    if TCP in pkt:
        protocol = "TCP"
        src_port = int(pkt[TCP].sport)
        dst_port = int(pkt[TCP].dport)
        # int(FlagValue) yields standard bits FIN=1 SYN=2 RST=4 PSH=8 ACK=16 URG=32,
        # matching the dpkt masks the offline flow builder uses.
        tcp_flags = int(pkt[TCP].flags)
    elif UDP in pkt:
        protocol = "UDP"
        src_port = int(pkt[UDP].sport)
        dst_port = int(pkt[UDP].dport)

    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": protocol,
        "timestamp": float(getattr(pkt, "time", getattr(pkt, "timestamp", 0.0))),
        "ip_len": ip_len,
        "tcp_flags": tcp_flags,
    }


def _extract_payload_bytes(pkt: Any) -> bytes:
    if isinstance(pkt, dict):
        raw = pkt.get("payload", pkt.get("payload_bytes", pkt.get("payload_raw", b"")))
        if isinstance(raw, str):
            return raw.encode("latin-1", errors="replace")
        if isinstance(raw, (bytes, bytearray, memoryview)):
            return bytes(raw)
        if isinstance(raw, np.ndarray):
            return raw.astype(np.uint8, copy=False).tobytes()
        if isinstance(raw, list):
            return bytes(int(x) & 0xFF for x in raw)
        return b""

    try:
        from scapy.packet import Raw
    except ImportError:
        raw = getattr(pkt, "payload", b"")
        return raw if isinstance(raw, bytes) else bytes(raw or b"")

    if Raw in pkt:
        return bytes(pkt[Raw].load)
    return b""


def _bytes_to_ascii(raw: bytes) -> str:
    return "".join(chr(byte) if 32 <= byte <= 126 else "." for byte in raw)


def _truncate_and_pad_payload(payload: bytes, payload_length: int) -> np.ndarray:
    fixed = np.zeros(int(payload_length), dtype=np.uint8)
    if not payload:
        return fixed
    clipped = np.frombuffer(payload[:payload_length], dtype=np.uint8)
    fixed[: clipped.shape[0]] = clipped
    return fixed
