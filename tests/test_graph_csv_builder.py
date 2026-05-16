import pandas as pd
import pytest

from graphslm_ids.offline.preprocessing.graph_csv_builder import build_graph_csv_tables


def test_graph_csv_builder_splits_flow_by_packet_cap() -> None:
    metadata = pd.DataFrame(
        [
            {
                "pcap_file": "sample.pcap",
                "packet_index": 0,
                "timestamp": 1.0,
                "label": "Benign",
                "src_ip": "10.0.0.1",
                "dst_ip": "10.0.0.2",
                "src_port": 1111,
                "dst_port": 80,
                "protocol": "TCP",
                "payload_len_raw": 40,
            },
            {
                "pcap_file": "sample.pcap",
                "packet_index": 1,
                "timestamp": 1.1,
                "label": "Benign",
                "src_ip": "10.0.0.1",
                "dst_ip": "10.0.0.2",
                "src_port": 1111,
                "dst_port": 80,
                "protocol": "TCP",
                "payload_len_raw": 35,
            },
            {
                "pcap_file": "sample.pcap",
                "packet_index": 2,
                "timestamp": 1.2,
                "label": "Benign",
                "src_ip": "10.0.0.1",
                "dst_ip": "10.0.0.2",
                "src_port": 1111,
                "dst_port": 80,
                "protocol": "TCP",
                "payload_len_raw": 20,
            },
        ]
    )

    tables = build_graph_csv_tables(
        metadata,
        flow_timeout_seconds=30.0,
        max_packets_per_flow=2,
    )

    assert tables.flow_nodes.shape[0] == 2
    assert tables.packet_nodes.shape[0] == 3
    assert tables.contain_edges.shape[0] == 3
    assert tables.link_edges.shape[0] == 1


def test_graph_csv_builder_requires_columns() -> None:
    bad_metadata = pd.DataFrame([{"timestamp": 1.0}])
    with pytest.raises(ValueError):
        build_graph_csv_tables(bad_metadata)
