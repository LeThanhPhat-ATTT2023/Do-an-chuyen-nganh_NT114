"""FL experiment configuration: defaults, YAML loading and dotted overrides."""

from __future__ import annotations

import copy
import re
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


# Subtrees whose keys are free-form (user-chosen names), so unknown_keys() does not police them.
OPEN_SUBTREES: tuple[str, ...] = (
    "experiment.labels",
    "attack.params",
    "defense.screen.params",
    "defense.aggregator.params",
    "sampler.fanouts",
    "sampler.reverse_fanouts",
    "eval.families",
)

_TIMESTAMP_TAG = "tag:yaml.org,2002:timestamp"
_FLOAT_TAG = "tag:yaml.org,2002:float"


class FLConfigLoader(yaml.SafeLoader):
    """SafeLoader tuned for configs: ``1e-3`` is a float and ``2026-10-02`` stays a string.

    PyYAML follows YAML 1.1, where a float needs a dot (``1e-3`` is read as the string
    ``'1e-3'``) and a bare date becomes ``datetime.date``. Both are surprising in a config file.
    The resolver tables below are copied per class, so the global ``yaml.SafeLoader`` is untouched.
    """


FLConfigLoader.yaml_implicit_resolvers = {
    first_char: [(tag, regexp) for tag, regexp in resolvers if tag != _TIMESTAMP_TAG]
    for first_char, resolvers in FLConfigLoader.yaml_implicit_resolvers.items()
}
FLConfigLoader.add_implicit_resolver(
    _FLOAT_TAG,
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def yaml_load(text_or_stream: Any) -> Any:
    """Parse YAML with :class:`FLConfigLoader` (safe; scientific floats, dates as strings)."""
    return yaml.load(text_or_stream, Loader=FLConfigLoader)  # SafeLoader subclass: safe


def deep_update(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if value is None and isinstance(out.get(key), dict):
            # An empty YAML section (``attack:``) must not wipe out the default sub-dict.
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_fl_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_FL_CONFIG)
    with Path(path).open("r", encoding="utf-8") as fh:
        user = yaml_load(fh) or {}
    if not isinstance(user, dict):
        raise ValueError(f"FL config {path} must be a YAML mapping")
    return deep_update(DEFAULT_FL_CONFIG, user)


def _in_open_subtree(path: str) -> bool:
    return any(path == open_path or path.startswith(open_path + ".") for open_path in OPEN_SUBTREES)


def unknown_keys(cfg: dict[str, Any], defaults: dict[str, Any] = DEFAULT_FL_CONFIG) -> list[str]:
    """Return sorted dotted paths in ``cfg`` that do not exist in ``defaults``.

    Free-form subtrees (``OPEN_SUBTREES``) are skipped entirely. Where the default value is not
    a dict (scalars, lists, ``None``), whatever ``cfg`` holds there is accepted and not descended.
    """
    found: list[str] = []

    def walk(node: dict[str, Any], reference: dict[str, Any], prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if _in_open_subtree(path):
                continue
            if key not in reference:
                found.append(path)
            elif isinstance(value, dict) and isinstance(reference[key], dict):
                walk(value, reference[key], path)

    walk(cfg, defaults, "")
    return sorted(found)


def _split_dotted(key: str) -> list[str]:
    parts = key.split(".")
    if any(not part for part in parts):
        raise ValueError(f"invalid dotted key {key!r}: empty key or empty segment")
    return parts


def set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = _split_dotted(key)
    node = config
    for depth, part in enumerate(parts[:-1]):
        child = node.get(part)
        if child is None:
            child = {}
            node[part] = child
        elif not isinstance(child, dict):
            where = ".".join(parts[: depth + 1])
            raise ValueError(
                f"cannot set {key!r}: {where!r} is a {type(child).__name__}, not a mapping"
            )
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
        key = key.strip()
        _split_dotted(key)
        out[key] = yaml_load(raw)
    return out


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Return a deep copy of ``config`` with dotted-key ``overrides`` applied.

    A dict value REPLACES the whole sub-dict at that key; it is not deep-merged the way a YAML
    file is in :func:`load_fl_config`. For example ``{"defense.aggregator": {"name": "median"}}``
    drops the default ``params`` entry, so consumers must read optional keys with
    ``.get(key, default)``. Use a deeper key (``defense.aggregator.name``) to change one leaf.
    """
    out = copy.deepcopy(config)
    for key, value in (overrides or {}).items():
        set_dotted(out, key, copy.deepcopy(value))
    return out
