"""Curated signature rules backing v2's evidence-grounded edges.

The intent is to capture the *kinds* of evidence each MITRE technique exhibits
in network traffic, sourced from two corpora:

  * **OWASP Core Rule Set (CRS)** — patterns extracted from REQUEST-942 (SQLi),
    REQUEST-941 (XSS), REQUEST-932 (Command Injection), REQUEST-933 (file
    upload / web shell). These are battle-tested across 15+ years of WAF
    deployment. We do NOT vendor the full CRS; we lift representative patterns
    and document the rule group they came from.
  * **Behavioural / flow-level signatures** — hand-curated rules over the v2
    flow features that capture scan/recon/flood behaviour the WAF rules can't
    see (because they don't look at flow-level fan-out / flag ratios).

Each rule maps to one MITRE ATT&CK technique id. Weights are in (0, 1] and
combined by the evidence-edge builder via sum + clip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class PayloadRule:
    id: str
    technique_id: str
    regex: str
    transformations: tuple[str, ...] = ()
    weight: float = 0.5
    provenance: str = ""


@dataclass(frozen=True)
class FlowRule:
    id: str
    technique_id: str
    # condition_fn(row) -> weight in [0, 1]; 0 means "did not fire"
    condition_fn: Callable
    provenance: str = ""


# --- Payload signatures -----------------------------------------------------
# CRS provenance comments reference the REQUEST-XXX rule group, not specific
# rule IDs (the IDs in CRS shift between minor versions; the GROUP is stable).

PAYLOAD_SIGNATURES: tuple[PayloadRule, ...] = (
    # ---- SQL injection (CRS REQUEST-942) -> T1190 Exploit Public-Facing App
    PayloadRule(
        id="sqli_union_select",
        technique_id="T1190",
        regex=r"union\s+(all\s+)?select\b",
        transformations=("lowercase", "url_decode"),
        weight=0.9,
        provenance="OWASP CRS REQUEST-942",
    ),
    PayloadRule(
        id="sqli_or_1_eq_1",
        technique_id="T1190",
        regex=r"or\s+1\s*=\s*1\b",
        transformations=("lowercase", "url_decode"),
        weight=0.8,
        provenance="OWASP CRS REQUEST-942",
    ),
    PayloadRule(
        id="sqli_comment_marker",
        technique_id="T1190",
        regex=r"--\s",
        transformations=("url_decode",),
        weight=0.4,
        provenance="OWASP CRS REQUEST-942",
    ),
    PayloadRule(
        id="sqli_info_schema",
        technique_id="T1190",
        regex=r"information_schema\b",
        transformations=("lowercase", "url_decode"),
        weight=0.7,
        provenance="OWASP CRS REQUEST-942",
    ),
    PayloadRule(
        id="sqli_sleep",
        technique_id="T1190",
        regex=r"\bsleep\s*\(",
        transformations=("lowercase", "url_decode"),
        weight=0.6,
        provenance="OWASP CRS REQUEST-942 (time-based blind)",
    ),
    PayloadRule(
        id="sqli_benchmark",
        technique_id="T1190",
        regex=r"\bbenchmark\s*\(",
        transformations=("lowercase", "url_decode"),
        weight=0.6,
        provenance="OWASP CRS REQUEST-942 (time-based blind)",
    ),
    PayloadRule(
        id="sqli_load_file",
        technique_id="T1190",
        regex=r"\bload_file\s*\(",
        transformations=("lowercase", "url_decode"),
        weight=0.7,
        provenance="OWASP CRS REQUEST-942",
    ),
    PayloadRule(
        id="sqli_into_outfile",
        technique_id="T1190",
        regex=r"\binto\s+outfile\b",
        transformations=("lowercase", "url_decode"),
        weight=0.8,
        provenance="OWASP CRS REQUEST-942",
    ),
    # ---- XSS / browser JS (CRS REQUEST-941) -> T1059.007 JavaScript
    PayloadRule(
        id="xss_script_tag",
        technique_id="T1059.007",
        regex=r"<script\b",
        transformations=("lowercase", "url_decode", "html_entity_decode"),
        weight=0.85,
        provenance="OWASP CRS REQUEST-941",
    ),
    PayloadRule(
        id="xss_onerror",
        technique_id="T1059.007",
        regex=r"\bonerror\s*=",
        transformations=("lowercase", "url_decode", "html_entity_decode"),
        weight=0.7,
        provenance="OWASP CRS REQUEST-941",
    ),
    PayloadRule(
        id="xss_onload",
        technique_id="T1059.007",
        regex=r"\bonload\s*=",
        transformations=("lowercase", "url_decode", "html_entity_decode"),
        weight=0.6,
        provenance="OWASP CRS REQUEST-941",
    ),
    PayloadRule(
        id="xss_javascript_proto",
        technique_id="T1059.007",
        regex=r"javascript:",
        transformations=("lowercase", "url_decode"),
        weight=0.7,
        provenance="OWASP CRS REQUEST-941",
    ),
    PayloadRule(
        id="xss_iframe",
        technique_id="T1059.007",
        regex=r"<iframe\b",
        transformations=("lowercase", "url_decode"),
        weight=0.5,
        provenance="OWASP CRS REQUEST-941",
    ),
    PayloadRule(
        id="xss_svg_onload",
        technique_id="T1059.007",
        regex=r"<svg\b[^>]*onload",
        transformations=("lowercase", "url_decode"),
        weight=0.7,
        provenance="OWASP CRS REQUEST-941",
    ),
    # ---- Shell command injection (CRS REQUEST-932) -> T1059.004 Unix Shell
    PayloadRule(
        id="cmd_etc_passwd",
        technique_id="T1059.004",
        regex=r"/etc/passwd\b",
        transformations=("lowercase", "url_decode"),
        weight=0.85,
        provenance="OWASP CRS REQUEST-932",
    ),
    PayloadRule(
        id="cmd_dollar_paren",
        technique_id="T1059.004",
        regex=r"\$\([^)]+\)",
        transformations=("url_decode",),
        weight=0.6,
        provenance="OWASP CRS REQUEST-932",
    ),
    PayloadRule(
        id="cmd_backtick",
        technique_id="T1059.004",
        regex=r"`[a-z0-9_\-]+`",
        transformations=("lowercase", "url_decode"),
        weight=0.55,
        provenance="OWASP CRS REQUEST-932",
    ),
    PayloadRule(
        id="cmd_pipe_sh",
        technique_id="T1059.004",
        regex=r"\|\s*(sh|bash|zsh)\b",
        transformations=("lowercase", "url_decode"),
        weight=0.7,
        provenance="OWASP CRS REQUEST-932",
    ),
    PayloadRule(
        id="cmd_wget_curl",
        technique_id="T1059.004",
        regex=r"\b(wget|curl)\s+https?://",
        transformations=("lowercase", "url_decode"),
        weight=0.5,
        provenance="OWASP CRS REQUEST-932",
    ),
    PayloadRule(
        id="cmd_bin_shell",
        technique_id="T1059.004",
        regex=r"/bin/(sh|bash)\b",
        transformations=("lowercase", "url_decode"),
        weight=0.7,
        provenance="OWASP CRS REQUEST-932",
    ),
    # ---- File upload anomalies (CRS REQUEST-933) -> T1505.003 Web Shell
    PayloadRule(
        id="upload_dangerous_ext",
        technique_id="T1505.003",
        regex=r"filename=[\"']?[^\"'\\\\/]*\.(php|jsp|asp|aspx|jspx|phtml)\b",
        transformations=("lowercase", "url_decode"),
        weight=0.85,
        provenance="OWASP CRS REQUEST-933",
    ),
    PayloadRule(
        id="upload_multipart_boundary",
        technique_id="T1505.003",
        regex=r"multipart/form-data;\s*boundary=",
        transformations=("lowercase",),
        weight=0.3,
        provenance="OWASP CRS REQUEST-933 (anomaly: high recall, low specificity)",
    ),
    PayloadRule(
        id="webshell_eval",
        technique_id="T1505.003",
        regex=r"\beval\s*\(",
        transformations=("lowercase", "url_decode"),
        weight=0.6,
        provenance="OWASP CRS REQUEST-933 (web shell exec)",
    ),
    PayloadRule(
        id="webshell_base64_decode",
        technique_id="T1505.003",
        regex=r"\bbase64_decode\s*\(",
        transformations=("lowercase", "url_decode"),
        weight=0.7,
        provenance="OWASP CRS REQUEST-933 (web shell exec)",
    ),
    PayloadRule(
        id="webshell_system",
        technique_id="T1505.003",
        regex=r"\b(system|passthru|shell_exec)\s*\(",
        transformations=("lowercase", "url_decode"),
        weight=0.7,
        provenance="OWASP CRS REQUEST-933 (web shell exec)",
    ),
)


# --- Flow signatures --------------------------------------------------------
# Each condition_fn returns a weight in [0, 1]. Reading flow rows safely is
# verbose because pandas Series.get() is the only way to tolerate missing
# columns when the same rule is applied to different flow snapshots in tests.


def _g(row, key: str, default: float = 0.0) -> float:
    try:
        v = row[key]
    except Exception:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _f_syn_fanout(row) -> float:
    """High SYN fan-out: a source contacts many dst ports with mostly SYN-only
    packets and tiny/empty payloads -> classic port scan (T1046)."""
    if _g(row, "proto") != 0:
        return 0.0
    if _g(row, "fan_dst_port") < 50:
        return 0.0
    if _g(row, "r_syn") < 0.7:
        return 0.0
    if _g(row, "plen_pay_mean") > 16:
        return 0.0
    return min(_g(row, "fan_dst_port") / 200.0, 1.0) * 0.5 + _g(row, "r_syn") * 0.5


def _f_host_fanout(row) -> float:
    """Source touches many distinct dst-IPs -> remote system discovery (T1018)."""
    if _g(row, "fan_dst_ip") < 50:
        return 0.0
    return min(_g(row, "fan_dst_ip") / 200.0, 1.0)


def _f_icmp_flood(row) -> float:
    if _g(row, "proto") != 2:
        return 0.0
    if _g(row, "tot_pkts") < 50:
        return 0.0
    if _g(row, "pkts_per_s") < 100:
        return 0.0
    return min(_g(row, "pkts_per_s") / 1000.0, 1.0)


def _f_icmp_pingsweep(row) -> float:
    if _g(row, "proto") != 2:
        return 0.0
    if _g(row, "fan_dst_ip") < 20:
        return 0.0
    return min(_g(row, "fan_dst_ip") / 100.0, 1.0)


def _f_syn_flood(row) -> float:
    if _g(row, "proto") != 0:
        return 0.0
    if _g(row, "r_syn") < 0.9:
        return 0.0
    if _g(row, "pkts_per_s") < 200:
        return 0.0
    return min(_g(row, "pkts_per_s") / 1000.0, 1.0)


def _f_rst_burst(row) -> float:
    if _g(row, "proto") != 0:
        return 0.0
    if _g(row, "r_rst") < 0.3:
        return 0.0
    if _g(row, "tot_pkts") < 20:
        return 0.0
    return min(_g(row, "r_rst"), 1.0)


def _f_slowloris(row) -> float:
    if _g(row, "proto") != 0:
        return 0.0
    if _g(row, "dur") < 30:
        return 0.0
    if _g(row, "r_psh") < 0.2:
        return 0.0
    if _g(row, "r_rst") > 0.05:
        return 0.0
    if _g(row, "pkts_per_s") > 5:
        return 0.0
    return 0.6


def _f_exfil_asymmetric(row) -> float:
    if _g(row, "proto") != 0:
        return 0.0
    if _g(row, "fwd_bytes") < 1_000_000:
        return 0.0
    if _g(row, "bwd_bytes") > 100_000:
        return 0.0
    return min(_g(row, "fwd_bytes") / 10_000_000.0, 1.0)


FLOW_SIGNATURES: tuple[FlowRule, ...] = (
    FlowRule("flow:syn_fanout", "T1046", _f_syn_fanout, "Network Service Scanning"),
    FlowRule("flow:host_fanout", "T1018", _f_host_fanout, "Remote System Discovery"),
    FlowRule("flow:icmp_flood", "T1498", _f_icmp_flood, "Network DoS"),
    FlowRule("flow:icmp_pingsweep", "T1018", _f_icmp_pingsweep, "Remote System Discovery"),
    FlowRule("flow:syn_flood", "T1498", _f_syn_flood, "Network DoS"),
    FlowRule("flow:rst_burst", "T1499", _f_rst_burst, "Endpoint DoS"),
    FlowRule("flow:slowloris", "T1499", _f_slowloris, "Endpoint DoS (slow)"),
    FlowRule(
        "flow:exfil_asymmetric",
        "T1041",
        _f_exfil_asymmetric,
        "Exfiltration over C2 Channel",
    ),
)


# Every technique id referenced by ANY rule -- the graph builder needs this list
# to guarantee the technique node exists even when no edge fires in a batch.
ALL_REFERENCED_TECHNIQUES: frozenset[str] = frozenset(
    {r.technique_id for r in PAYLOAD_SIGNATURES}
    | {r.technique_id for r in FLOW_SIGNATURES}
)
