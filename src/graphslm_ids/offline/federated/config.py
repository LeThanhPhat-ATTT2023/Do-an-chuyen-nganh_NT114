"""FL experiment configuration: defaults, YAML loading and dotted overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_FAMILIES: dict[str, list[str]] = {
    "web": ["CommandInjection", "XSS", "SqlInjection", "Uploading_Attack", "BrowserHijacking"],
    "ddos": [
        "DDoS-ACK_Fragmentation",
        "DDoS-ICMP_Flood",
        "DDoS-ICMP_Fragmentation",
        "DDoS-PSHACK_Flood",
        "DDoS-RSTFINFlood",
    ],
    "recon": [
        "Recon-HostDiscovery",
        "Recon-OSScan",
        "Recon-PingSweep",
        "Recon-PortScan",
        "VulnerabilityScan",
    ],
    "mirai": ["Mirai-udpplain"],
    "malware": ["Backdoor_Malware"],
}

DEFAULT_FL_CONFIG: dict[str, Any] = {
    "experiment": {"name": "fl_run", "labels": {}},
    "data": {
        "artifact_version": "v3",
        "graph_npz": "outputs/v3_ob/graph.npz",
        "graph_meta_json": "outputs/v3_ob/graph.meta.json",
        "splits_json": "outputs/v3_ob/splits.json",
        "split_protocol": "random",
        "packet_dtype": "preserve",
        "add_reverse_edges": True,
        "use_semantic_edge_weights": True,
        "standardize_flow_features": True,
        "clean_eval_labels": None,
    },
    "model": {
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.2,
        "ffn_multiplier": 2,
    },
    "sampler": {
        "hops": 2,
        "fanouts": {},
        "reverse_fanouts": {},
        "always_include_all_techniques": True,
        "always_include_all_tactics": True,
    },
    "partition": {
        "scheme": "dirichlet",
        "num_clients": 10,
        "alpha": 0.5,
        "min_flows_per_client": 50,
        "max_flows_per_client": None,
        "seed": 42,
    },
    "client": {
        "local_epochs": 1,
        "max_local_steps": None,
        "batch_seed_flows": 256,
        "lr": 1.0e-3,
        "lr_schedule": "cosine_rounds",
        "lr_min": 1.0e-5,
        "weight_decay": 5.0e-5,
        "adamw_eps": 1.0e-6,
        "adamw_betas": [0.9, 0.95],
        "grad_clip_norm": 1.0,
        "optimizer_state": "reset",
        "loss_type": "focal",
        "focal_gamma": 1.5,
        "label_smoothing": 0.05,
        "class_weight": "balanced",
        "class_weight_method": "cb",
        "cb_beta": 0.985,
        "class_weight_cap": 4.0,
        "stratified_batch_sampling": True,
        "drop_edge_prob": 0.1,
        "num_workers": 0,
        "pin_memory": False,
        "gcl": {
            "enabled": False,
            "weight": 0.3,
            "temperature": 0.1,
            "n_negatives": 16,
            "class_technique_map_csv": "data/mitre/class_technique_map.csv",
        },
        "eacs": {
            "enabled": False,
            "anchor_mask_npy": None,
            "warmup_rounds": 5,
            "ema_decay": 0.9,
            "lambda_disambig": 0.7,
            "suspect_classes": ["CommandInjection", "XSS", "SqlInjection", "Uploading_Attack"],
            "mitre_dir": "data/mitre",
        },
    },
    "federated": {
        "mode": "fl",
        "rounds": 30,
        "participation": 1.0,
        "server_lr": 1.0,
        "seed": 42,
        "cache_clients": True,
    },
    "attack": {
        "name": "none",
        "malicious_ratio": 0.0,
        "knowledge": "omniscient",
        "start_round": 1,
        "params": {},
    },
    "defense": {
        "clip": {"mode": "none", "factor": 1.0, "value": None},
        "screen": {"name": "none", "params": {}},
        "aggregator": {"name": "fedavg", "params": {}},
        "assumed_malicious_ratio": 0.2,
        "relation_aware": False,
        "weighting": "samples",
    },
    "eval": {
        "every": 1,
        "max_val_flows": None,
        "monitor": "val_macro_f1",
        "batch_seed_flows": 512,
        "families": copy.deepcopy(DEFAULT_FAMILIES),
    },
    "output": {"dir": "outputs/fl/fl_run", "save_checkpoints": True},
    "device": "auto",
}


def deep_update(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_fl_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_FL_CONFIG)
    with Path(path).open("r", encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    if not isinstance(user, dict):
        raise ValueError(f"FL config {path} must be a YAML mapping")
    return deep_update(DEFAULT_FL_CONFIG, user)


def set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def get_dotted(config: dict[str, Any], key: str, default: Any = None) -> Any:
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def parse_set_args(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(raw)
    return out


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(config)
    for key, value in (overrides or {}).items():
        set_dotted(out, key, copy.deepcopy(value))
    return out
