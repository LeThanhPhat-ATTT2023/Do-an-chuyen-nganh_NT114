import numpy as np
import pandas as pd

from graphslm_ids.offline_path.preprocessing import build_three_tier_graph_artifact as module


def test_select_top_k_above_threshold() -> None:
    similarity = np.array(
        [
            [0.91, 0.88, 0.10],
            [0.40, 0.95, 0.94],
        ],
        dtype=np.float32,
    )

    edges = module._select_top_k_above_threshold(similarity, top_k=2, threshold=0.9)

    assert (0, 0, 0.91) in [(src, dst, round(score, 2)) for src, dst, score in edges]
    rounded = {(src, dst, round(score, 2)) for src, dst, score in edges}
    assert rounded == {(0, 0, 0.91), (1, 1, 0.95), (1, 2, 0.94)}


def test_build_technique_tactic_arrays() -> None:
    techniques_df = pd.DataFrame(
        [
            {"technique_id": "T1001", "name": "Technique A"},
            {"technique_id": "T1002", "name": "Technique B"},
        ]
    )
    edges_df = pd.DataFrame(
        [
            {"technique_id": "T1001", "tactic_shortname": "initial-access"},
            {"technique_id": "T1002", "tactic_shortname": "execution"},
            {"technique_id": "T9999", "tactic_shortname": "execution"},
        ]
    )

    edge_index, edge_attr, technique_map, tactic_map = module._build_technique_tactic_arrays(
        techniques_df, edges_df
    )

    assert technique_map == {"T1001": 0, "T1002": 1}
    assert set(tactic_map.keys()) == {"execution", "initial-access"}
    assert edge_index.shape == (2, 2)
    assert edge_attr.shape == (2, 1)
    assert np.allclose(edge_attr, np.ones((2, 1), dtype=np.float32))
