import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# nfstream column-name variants across versions
_PROTO_CANDIDATES = ["protocol", "transport_protocol", "ip_protocol", "Protocol"]
_PKT_SIZE_COLS = ["src2dst_min_ps", "src2dst_max_ps", "dst2src_min_ps", "dst2src_max_ps"]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def additional_features(
    file_name,
    window_size=350,
    http_ports=[443, 8080, 80],
    vulnerable_ports=[20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3389, 8080],
    dns_ports=[53],
    exp_id=[0, -1],
    proto_list=[1, 2, 6, 17, 58],
):
    """Compute rolling-window features on a nfstream CSV and overwrite it in-place."""
    try:
        data = pd.read_csv(file_name)
    except Exception:
        print(f"file reading error: {file_name}")
        return ""
    try:
        data = data.sort_values(by="bidirectional_first_seen_ms")
    except Exception:
        print("CSV does not contain bidirectional_first_seen_ms for initial sorting")
        return ""

    # Defragment after many-column nfstream output
    data = data.copy()

    def rolling_sum(group, w):
        return group.rolling(window=w, min_periods=1).sum()

    def rolling_mean(group, w):
        return group.rolling(window=w, min_periods=1).mean()

    def rolling_unique(group, w):
        return group.rolling(window=w, min_periods=1).apply(
            lambda x: len(set(x)), raw=True
        )

    le1, le2 = LabelEncoder(), LabelEncoder()
    data["src_dst_ip"] = data["src_ip"].astype(str) + "-" + data["dst_ip"].astype(str)
    data["src_dst_encoded"] = le1.fit_transform(data["src_dst_ip"])
    data["dst_ip_encoded"] = le2.fit_transform(data["dst_ip"].astype(str))

    # packet_size_variation — skip gracefully if nfstream omits ps columns
    ps_cols_present = [c for c in _PKT_SIZE_COLS if c in data.columns]
    if len(ps_cols_present) == 4:
        data["packet_size_variation"] = data[_PKT_SIZE_COLS].std(axis=1)
    else:
        data["packet_size_variation"] = 0.0

    # Detect transport-protocol column (name varies across nfstream versions)
    proto_col = _find_col(data, _PROTO_CANDIDATES)
    if proto_col is None:
        print(
            f"[additional_features] WARNING: no protocol column found in {file_name}.\n"
            f"  Available columns: {list(data.columns)}\n"
            "  Protocol-based features will be zero."
        )
        data["is_udp_request"] = 0
        data["is_tcp_request"] = 0
        data["is_icmp_request"] = 0
    else:
        data["is_udp_request"] = (data[proto_col] == 17).astype(int)
        data["is_tcp_request"] = (data[proto_col] == 6).astype(int)
        data["is_icmp_request"] = (data[proto_col] == 1).astype(int)

    for col, feat, label in [
        ("Rolling_UDP_Requests_SourceDestination", "is_udp_request", "src_dst_encoded"),
        ("Rolling_UDP_Requests_Destination", "is_udp_request", "dst_ip_encoded"),
        ("Rolling_TCP_Requests_SourceDestination", "is_tcp_request", "src_dst_encoded"),
        ("Rolling_TCP_Requests_Destination", "is_tcp_request", "dst_ip_encoded"),
        ("Rolling_ACK_Packets_SourceDestination", "bidirectional_ack_packets", "src_dst_encoded"),
        ("Rolling_ACK_Packets_Destination", "bidirectional_ack_packets", "dst_ip_encoded"),
        ("Rolling_FIN_Packets_SourceDestination", "bidirectional_fin_packets", "src_dst_encoded"),
        ("Rolling_FIN_Packets_Destination", "bidirectional_fin_packets", "dst_ip_encoded"),
        ("Rolling_rst_Packets_SourceDestination", "bidirectional_rst_packets", "src_dst_encoded"),
        ("Rolling_rst_Packets_Destination", "bidirectional_rst_packets", "dst_ip_encoded"),
        ("Rolling_psh_Packets_SourceDestination", "bidirectional_psh_packets", "src_dst_encoded"),
        ("Rolling_psh_Packets_Destination", "bidirectional_psh_packets", "dst_ip_encoded"),
        ("Rolling_SYN_Packets_SourceDestination", "bidirectional_syn_packets", "src_dst_encoded"),
        ("Rolling_SYN_Packets_Destination", "bidirectional_syn_packets", "dst_ip_encoded"),
        ("Rolling_ICMP_Requests_SourceDestination", "is_icmp_request", "src_dst_encoded"),
        ("Rolling_ICMP_Requests_Destination", "is_icmp_request", "dst_ip_encoded"),
    ]:
        if feat not in data.columns:
            data[col] = 0.0
            continue
        data[col] = (
            data.groupby(label)[feat]
            .apply(lambda x: rolling_sum(x, window_size))
            .reset_index(level=0, drop=True)
        )

    data["Unique_Ports_In_SourceDestinationIP"] = (
        data.groupby("src_dst_encoded")["dst_port"]
        .apply(lambda x: rolling_unique(x, window_size))
        .reset_index(level=0, drop=True)
    )

    for col, label in [
        ("Rolling_Duration_Destination", "dst_ip_encoded"),
        ("Rolling_Duration_SourceDestination", "src_dst_encoded"),
    ]:
        data[col] = (
            data.groupby(label)["bidirectional_duration_ms"]
            .apply(lambda x: rolling_mean(x, window_size))
            .reset_index(level=0, drop=True)
        )

    for port_set, prefix in [(http_ports, "http"), (dns_ports, "dns"), (dns_ports, "dns2")]:
        data["is_vulnerable_port"] = data["dst_port"].isin(port_set).astype(int)
        data[f"Rolling_{prefix}_port_SourceDestination"] = (
            data.groupby("src_dst_encoded")["is_vulnerable_port"]
            .apply(lambda x: rolling_sum(x, window_size))
            .reset_index(level=0, drop=True)
        )
        data[f"Rolling_{prefix}_port_Destination"] = (
            data.groupby("dst_ip_encoded")["is_vulnerable_port"]
            .apply(lambda x: rolling_sum(x, window_size))
            .reset_index(level=0, drop=True)
        )

    data["is_vulnerable_port"] = data["dst_port"].isin(vulnerable_ports).astype(int)
    data["Rolling_vulnerable_port"] = (
        data.groupby("src_dst_encoded")["is_vulnerable_port"]
        .apply(lambda x: rolling_sum(x, window_size))
        .reset_index(level=0, drop=True)
    )
    data["Rolling_packets_destination"] = (
        data.groupby("dst_ip_encoded")["src2dst_packets"]
        .apply(lambda x: rolling_sum(x, window_size))
        .reset_index(level=0, drop=True)
    )
    data["Rolling_bipackets_destination"] = (
        data.groupby("dst_ip_encoded")["bidirectional_packets"]
        .apply(lambda x: rolling_sum(x, window_size))
        .reset_index(level=0, drop=True)
    )

    data["expiration_id"] = pd.Categorical(data["expiration_id"], categories=exp_id)
    dummy_cols = ["expiration_id"]
    dummy_prefixes = ["Exp"]
    if proto_col is not None and proto_col in data.columns:
        data[proto_col] = pd.Categorical(data[proto_col], categories=proto_list)
        dummy_cols.append(proto_col)
        dummy_prefixes.append("proto")
    data = pd.get_dummies(
        data, prefix=dummy_prefixes, columns=dummy_cols, dtype=int
    )
    data.drop(
        ["src_dst_ip", "src_dst_encoded", "dst_ip_encoded",
         "is_udp_request", "is_tcp_request", "is_icmp_request", "is_vulnerable_port"],
        axis=1, inplace=True, errors="ignore",
    )
    data.to_csv(file_name, index=False)
    return file_name
