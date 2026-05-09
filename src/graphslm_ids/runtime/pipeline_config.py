from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from graphslm_ids.slow_path.report_generator import ReportGeneratorConfig
from graphslm_ids.slow_path.report_validator import ValidatorConfig
from graphslm_ids.slow_path.slow_path_worker import SlowPathConfig


@dataclass
class FastPathCfg:
    student_onnx: str
    mitre_embeddings: str
    techniques_csv: str
    technique_tactic_csv: str
    mitre_topk: int = 5
    mitre_threshold: float | None = None
    payload_length: int = 256


@dataclass
class HotGraphCfg:
    ttl_seconds: float = 60.0
    max_events: int = 100_000
    max_packets_per_flow: int = 64
    max_techniques_per_node: int = 5


@dataclass
class HGTCfg:
    checkpoint: str
    graph_meta_json: str
    device: str = "cpu"
    num_layers: int = 3
    packet_feature: str = "semantic"
    add_reverse_edges: bool = True
    standardize_flow_features: bool = True
    use_semantic_edge_weights: bool = True


@dataclass
class PolicyCfg:
    alert_threshold: float = 0.70
    alert_labels: tuple[str, ...] = ("suspicious", "malicious")


@dataclass
class PipelineConfig:
    fast_path: FastPathCfg
    hot_graph: HotGraphCfg
    hgt: HGTCfg
    policy: PolicyCfg
    slow_path: SlowPathConfig
    slm: ReportGeneratorConfig
    validator: ValidatorConfig
    cold_store_path: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError("Pipeline config must be a YAML mapping.")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "PipelineConfig":
        fast_raw = dict(raw.get("fast_path") or {})
        data_raw = dict(raw.get("data") or {})
        mitre_raw = dict(raw.get("mitre") or {})
        hgt_raw = dict(raw.get("hgt_runtime") or raw.get("hgt") or {})
        slow_raw = dict(raw.get("slow_path") or {})
        slm_raw = dict(raw.get("slm") or {})
        validator_raw = dict(raw.get("validator") or {})
        policy_raw = dict(raw.get("policy") or {})
        hot_raw = dict(raw.get("hot_graph") or {})
        cold_raw = raw.get("cold_store")

        hgt_output_dir = hgt_raw.get("output_dir", "outputs/hgt_flow_classifier_t082_k5_l3_d01")
        checkpoint = hgt_raw.get("checkpoint") or str(Path(hgt_output_dir) / "hgt_flow_best.pt")

        fast_path = FastPathCfg(
            student_onnx=str(fast_raw.get("student_onnx", "outputs/student_cnn/student_cnn.onnx")),
            mitre_embeddings=str(
                fast_raw.get("mitre_embeddings", mitre_raw.get("embedding_path", "data/mitre/mitre_techniques_embeddings.npy"))
            ),
            techniques_csv=str(
                fast_raw.get("techniques_csv", mitre_raw.get("techniques_csv", "data/mitre/mitre_techniques.csv"))
            ),
            technique_tactic_csv=str(
                fast_raw.get(
                    "technique_tactic_csv",
                    mitre_raw.get("technique_tactic_csv", "data/mitre/mitre_technique_tactic_edges.csv"),
                )
            ),
            mitre_topk=int(fast_raw.get("mitre_topk", mitre_raw.get("packet_top_k", 5))),
            mitre_threshold=_optional_float(
                fast_raw.get("mitre_threshold", mitre_raw.get("similarity_threshold"))
            ),
            payload_length=int(fast_raw.get("payload_length", data_raw.get("payload_length", 256))),
        )

        hgt = HGTCfg(
            checkpoint=str(checkpoint),
            graph_meta_json=str(
                hgt_raw.get("graph_meta_json", "data/processed/graph_artifact_3tier_t082_k5.meta.json")
            ),
            device=str(hgt_raw.get("device", "cpu")),
            num_layers=int(hgt_raw.get("num_layers", 3)),
            packet_feature=str(hgt_raw.get("packet_feature", "semantic")),
            add_reverse_edges=bool(hgt_raw.get("add_reverse_edges", True)),
            standardize_flow_features=bool(hgt_raw.get("standardize_flow_features", True)),
            use_semantic_edge_weights=bool(hgt_raw.get("use_semantic_edge_weights", True)),
        )

        hot_graph = HotGraphCfg(
            ttl_seconds=float(hot_raw.get("ttl_seconds", 60.0)),
            max_events=int(hot_raw.get("max_events", 100_000)),
            max_packets_per_flow=int(
                hot_raw.get(
                    "max_packets_per_flow",
                    hgt_raw.get("max_packets_per_flow", 64),
                )
            ),
            max_techniques_per_node=int(
                hot_raw.get("max_techniques_per_node", mitre_raw.get("packet_top_k", 5))
            ),
        )

        policy = PolicyCfg(
            alert_threshold=float(policy_raw.get("alert_threshold", slow_raw.get("alert_threshold", 0.70))),
            alert_labels=tuple(str(x) for x in policy_raw.get("alert_labels", ["suspicious", "malicious"])),
        )

        slow_path = SlowPathConfig(
            queue_timeout=float(slow_raw.get("queue_timeout", 1.0)),
            max_cf_packets=int(slow_raw.get("max_cf_packets", 10)),
            top_packets=int(slow_raw.get("top_packets", 5)),
            top_techniques=int(slow_raw.get("top_techniques", 5)),
            top_paths=int(slow_raw.get("top_paths", 5)),
            alert_threshold=float(policy.alert_threshold),
            max_payload_preview_bytes=int(slow_raw.get("max_payload_preview_bytes", 64)),
        )
        setattr(slow_path, "queue_max_size", int(slow_raw.get("queue_max_size", 1000)))

        slm = ReportGeneratorConfig(
            backend=slm_raw.get("backend", "ollama"),
            endpoint=str(slm_raw.get("endpoint", "http://localhost:11434")),
            model_name=slm_raw.get("model", slm_raw.get("model_name", "qwen2.5:3b-instruct-q4_k_m")),
            temperature=float(slm_raw.get("temperature", 0.15)),
            top_p=float(slm_raw.get("top_p", 0.80)),
            repeat_penalty=float(slm_raw.get("repeat_penalty", 1.10)),
            context_length=int(slm_raw.get("context_length", 8192)),
            max_new_tokens=int(slm_raw.get("max_new_tokens", 1500)),
            tier2_max_new_tokens=int(slm_raw.get("tier2_max_new_tokens", 800)),
            timeout_seconds=int(slm_raw.get("timeout_seconds", 30)),
        )
        validator = ValidatorConfig(
            grounding_threshold=float(validator_raw.get("grounding_threshold", 0.60)),
            max_hallucinated_entities=int(validator_raw.get("max_hallucinated_entities", 2)),
            max_unsupported_claims=int(validator_raw.get("max_unsupported_claims", 0)),
            require_mitre_caution=bool(validator_raw.get("require_mitre_caution", True)),
        )

        if isinstance(cold_raw, dict):
            cold_store_path = str(cold_raw.get("path", slow_raw.get("cold_store", "data/runtime/events.jsonl")))
        elif isinstance(cold_raw, str):
            cold_store_path = cold_raw
        else:
            cold_store_path = str(slow_raw.get("cold_store", "data/runtime/events.jsonl"))

        return PipelineConfig(
            fast_path=fast_path,
            hot_graph=hot_graph,
            hgt=hgt,
            policy=policy,
            slow_path=slow_path,
            slm=slm,
            validator=validator,
            cold_store_path=cold_store_path,
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
