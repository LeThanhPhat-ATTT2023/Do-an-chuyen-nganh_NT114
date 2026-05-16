from __future__ import annotations

from typing import Any, Iterator


class HGTCounterfactual:
    """Thin model-compatible wrapper used by EvidenceBuilder for packet masking."""

    def __init__(self, hgt_runtime: Any, max_cf_packets: int = 10) -> None:
        self.hgt_runtime = hgt_runtime
        self.model = hgt_runtime.model
        self.max_cf_packets = int(max_cf_packets)
        self.edge_types = list(getattr(self.model, "edge_types", []))

    def eval(self) -> "HGTCounterfactual":
        self.model.eval()
        return self

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        self.model.zero_grad(*args, **kwargs)

    def parameters(self) -> Iterator[Any]:
        return self.model.parameters()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)
