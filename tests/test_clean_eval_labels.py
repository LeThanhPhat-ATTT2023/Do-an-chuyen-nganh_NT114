"""Keying + relabel logic of the clean eval answer key (pure, no pcaps)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# scripts/tools is not an installed package; make it importable.
_TOOLS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from extract_clean_eval_labels import (  # noqa: E402
    canonical_key_of_flow_id,
    clean_labels_from_attack_keys,
    label_of_flow_id,
)

LABEL_MAPPING = {"Benign": 1, "CommandInjection": 3, "XSS": 17, "Recon-OSScan": 11}


def test_flow_id_parsing():
    fid = "CommandInjection|10.0.0.1:80|10.0.0.2:5555|6#2.1"
    assert label_of_flow_id(fid) == "CommandInjection"
    assert canonical_key_of_flow_id(fid) == "10.0.0.1:80|10.0.0.2:5555|6"


def test_web_flow_with_attack_key_keeps_label():
    order = ["CommandInjection|a:1|b:2|6#1.0"]
    keys = {"CommandInjection": {"a:1|b:2|6"}}
    out, audit = clean_labels_from_attack_keys(order, LABEL_MAPPING, keys)
    assert out.tolist() == [3]
    assert audit == {}


def test_web_flow_without_attack_key_demoted_to_benign():
    order = ["CommandInjection|a:1|b:2|6#1.0", "XSS|c:3|d:4|6#1.0"]
    keys = {"CommandInjection": set(), "XSS": set()}
    out, audit = clean_labels_from_attack_keys(order, LABEL_MAPPING, keys)
    assert out.tolist() == [1, 1]
    assert audit == {"CommandInjection": 1, "XSS": 1}


def test_non_web_classes_untouched():
    order = ["Recon-OSScan|a:1|b:2|6#1.0", "Benign|x:1|y:2|17#1.0"]
    out, audit = clean_labels_from_attack_keys(order, LABEL_MAPPING, {})
    assert out.tolist() == [11, 1]
    assert audit == {}


def test_segment_suffix_does_not_leak_into_key():
    # multiple segments of the same 5-tuple share the canonical key
    order = [
        "XSS|a:1|b:2|6#1.0",
        "XSS|a:1|b:2|6#2.0",
    ]
    keys = {"XSS": {"a:1|b:2|6"}}
    out, _ = clean_labels_from_attack_keys(order, LABEL_MAPPING, keys)
    assert out.tolist() == [17, 17]


def test_dtype_is_int64():
    out, _ = clean_labels_from_attack_keys(
        ["Benign|x:1|y:2|17#1.0"], LABEL_MAPPING, {}
    )
    assert out.dtype == np.int64
