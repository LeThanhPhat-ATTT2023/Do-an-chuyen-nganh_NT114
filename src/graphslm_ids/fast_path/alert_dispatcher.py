from __future__ import annotations

import queue
import time
import uuid
from typing import Any

from graphslm_ids.slow_path.types import SlowPathJob


class AlertDispatcher:
    """Create SlowPathJob objects and enqueue them without blocking the fast path."""

    def __init__(
        self,
        slow_queue: Any,
        cold_store: Any | None,
        alert_id_prefix: str = "alert",
    ) -> None:
        self.slow_queue = slow_queue
        self.cold_store = cold_store
        self.alert_id_prefix = alert_id_prefix
        self.dropped_alerts = 0

    def dispatch(
        self,
        decision: Any,
        flow_id: str,
        buffer: Any,
        subgraph: Any,
        attention: dict[str, float],
        timestamp: float | None = None,
    ) -> str | None:
        if not decision.should_alert:
            return None

        alert_id = self._gen_alert_id()
        snapshot = buffer.snapshot(flow_id)
        graph_snapshot = (
            subgraph.to_snapshot_dict() if hasattr(subgraph, "to_snapshot_dict") else dict(subgraph)
        )
        snapshot["graph_subgraph"] = graph_snapshot
        if self.cold_store is not None:
            self.cold_store.append_alert_snapshot(alert_id, flow_id, snapshot)

        job = SlowPathJob(
            alert_id=alert_id,
            flow_id=flow_id,
            predicted_label=decision.label,
            confidence=float(decision.confidence),
            subgraph_snapshot=graph_snapshot,
            hgt_attention={"packet_attention": dict(attention)},
            timestamp=float(timestamp if timestamp is not None else time.time()),
            top_classes=list(decision.top_classes),
            alert_threshold=float(decision.alert_threshold),
            predicted_label_idx=int(decision.label_idx),
        )

        try:
            self.slow_queue.put_nowait(job)
        except queue.Full:
            self.dropped_alerts += 1
            return None
        return alert_id

    def _gen_alert_id(self) -> str:
        return f"{self.alert_id_prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
