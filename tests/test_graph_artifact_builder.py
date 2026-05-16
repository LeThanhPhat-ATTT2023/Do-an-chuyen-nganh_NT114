import numpy as np
import pandas as pd

from graphslm_ids.offline.preprocessing.graph_artifact_builder import build_graph_artifact


def test_build_graph_artifact_shapes() -> None:
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
                "payload_len_raw": 20,
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
                "payload_len_raw": 12,
            },
        ]
    )
    payload = np.zeros((2, 256), dtype=np.uint8)
    payload[0, :3] = np.array([1, 2, 3], dtype=np.uint8)
    payload[1, :2] = np.array([9, 8], dtype=np.uint8)

    artifact = build_graph_artifact(
        metadata=metadata,
        payload_matrix=payload,
        flow_timeout_seconds=30.0,
        max_packets_per_flow=20,
    )

    assert artifact.arrays["flow_x"].shape[0] == 1
    assert artifact.arrays["packet_x"].shape == (2, 256)
    assert artifact.arrays["contain_edge_index"].shape == (2, 2)
    assert artifact.arrays["link_edge_index"].shape == (2, 1)
    assert artifact.arrays["link_edge_attr"].shape == (1, 1)
