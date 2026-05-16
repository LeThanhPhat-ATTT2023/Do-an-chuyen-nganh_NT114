from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import uuid


@dataclass(frozen=True)
class FlowKey:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str


@dataclass
class FlowState:
    flow_id: str
    first_seen: float
    last_seen: float
    packet_count: int
    key: FlowKey


class FlowTracker:
    """Map packets to stable flow IDs within an idle timeout window."""

    def __init__(self, idle_timeout_seconds: float = 60.0, flow_id_prefix: str = "flow") -> None:
        self.idle_timeout = float(idle_timeout_seconds)
        self.flow_id_prefix = flow_id_prefix
        self._table: dict[FlowKey, FlowState] = {}
        self._counter = 0

    def update(self, pkt: Any, now: float) -> FlowState:
        key = self._make_key(pkt)
        state = self._table.get(key)
        if state is None or (float(now) - state.last_seen) > self.idle_timeout:
            state = FlowState(
                flow_id=self._gen_id(),
                first_seen=float(now),
                last_seen=float(now),
                packet_count=0,
                key=key,
            )
            self._table[key] = state

        state.last_seen = float(now)
        state.packet_count += 1
        return state

    def evict_idle(self, now: float) -> list[str]:
        evicted: list[str] = []
        for key, state in list(self._table.items()):
            if (float(now) - state.last_seen) > self.idle_timeout:
                evicted.append(state.flow_id)
                del self._table[key]
        return evicted

    def __len__(self) -> int:
        return len(self._table)

    def _gen_id(self) -> str:
        self._counter += 1
        suffix = uuid.uuid4().hex[:8]
        return f"{self.flow_id_prefix}_{self._counter:08d}_{suffix}"

    def _make_key(self, pkt: Any) -> FlowKey:
        src_ip = _field(pkt, "src_ip", "src")
        dst_ip = _field(pkt, "dst_ip", "dst")
        src_port = _field(pkt, "src_port", "sport", default=0)
        dst_port = _field(pkt, "dst_port", "dport", default=0)
        protocol = _field(pkt, "protocol", "proto", default="OTHER")
        return FlowKey(
            src_ip=str(src_ip),
            dst_ip=str(dst_ip),
            src_port=int(src_port or 0),
            dst_port=int(dst_port or 0),
            protocol=str(protocol or "OTHER").upper(),
        )


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return default

    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)

    try:
        from scapy.layers.inet import IP, TCP, UDP
    except ImportError:
        return default

    try:
        if IP in obj:
            ip_layer = obj[IP]
            if "src_ip" in names or "src" in names:
                return ip_layer.src
            if "dst_ip" in names or "dst" in names:
                return ip_layer.dst
            if "protocol" in names or "proto" in names:
                if TCP in obj:
                    return "TCP"
                if UDP in obj:
                    return "UDP"
                return str(getattr(ip_layer, "proto", "OTHER"))
        if TCP in obj:
            if "src_port" in names or "sport" in names:
                return int(obj[TCP].sport)
            if "dst_port" in names or "dport" in names:
                return int(obj[TCP].dport)
        if UDP in obj:
            if "src_port" in names or "sport" in names:
                return int(obj[UDP].sport)
            if "dst_port" in names or "dport" in names:
                return int(obj[UDP].dport)
    except Exception:
        return default

    return default
