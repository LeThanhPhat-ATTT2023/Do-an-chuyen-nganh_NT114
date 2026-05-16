from __future__ import annotations

from typing import Protocol

from graphslm_ids.runtime.slow_path.types import GraphContext


class ContextSource(Protocol):
    def get_context(self, flow_id: str) -> GraphContext | None:
        ...


class ColdStoreSource(Protocol):
    def load_context(self, flow_id: str) -> GraphContext | None:
        ...


class ContextHydrator:
    def hydrate(
        self,
        flow_id: str,
        hot_buffer: ContextSource | None = None,
        cold_store: ColdStoreSource | None = None,
    ) -> GraphContext:
        if cold_store is not None and getattr(cold_store, "source_of_truth", False):
            context = cold_store.load_context(flow_id)
            if context is not None:
                return context

        if hot_buffer is not None:
            context = hot_buffer.get_context(flow_id)
            if context is not None:
                return context

        if cold_store is not None:
            context = cold_store.load_context(flow_id)
            if context is not None:
                return context

        raise KeyError(f"Context not found for flow_id={flow_id}")
