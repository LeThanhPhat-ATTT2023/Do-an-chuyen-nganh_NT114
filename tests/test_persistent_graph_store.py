from __future__ import annotations

import numpy as np

from graphslm_ids.fast_path import HotGraphBuffer, MitreIndex, SubgraphBuilder
from graphslm_ids.offline_path.training.hetero_graph_artifact import load_graph_store_artifact
from graphslm_ids.runtime import PersistentGraphStore
from graphslm_ids.slow_path import HotBufferAdapter
from graphslm_ids.slow_path.context_hydrator import ContextHydrator


def _mitre_index() -> MitreIndex:
    embeddings = np.eye(2, 4, dtype=np.float32)
    return MitreIndex.from_arrays(
        embeddings,
        [
            {"technique_id": "T1001", "name": "Technique A"},
            {"technique_id": "T1002", "name": "Technique B"},
        ],
        [
            {"technique_id": "T1001", "tactic_shortname": "initial-access"},
            {"technique_id": "T1002", "tactic_shortname": "execution"},
        ],
    )


def _append_packet(store: PersistentGraphStore, *, payload_ascii: str = "GET ") -> None:
    store.append_packet(
        packet_id="pkt_1",
        flow_id="flow_1",
        embedding=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        payload_hex=payload_ascii.encode("ascii").hex(),
        payload_ascii=payload_ascii,
        payload_len_raw=len(payload_ascii),
        timestamp=1.0,
        src_ip="192.168.1.10",
        dst_ip="10.0.0.5",
        src_port=51522,
        dst_port=80,
        protocol="TCP",
        mitre_topk=[("T1001", 0.91)],
        flow_label="malicious",
    )


def test_persistent_graph_store_round_trip_context(tmp_path) -> None:
    root = tmp_path / "graph_store"
    store = PersistentGraphStore(root, mitre_index=_mitre_index(), packet_embedding_dim=4)
    _append_packet(store)

    reloaded = PersistentGraphStore(root, packet_embedding_dim=4)
    context = reloaded.load_context("flow_1")

    assert context is not None
    assert context.flow.flow_id == "flow_1"
    assert context.flow.packet_count == 1
    assert context.packets[0].payload_preview_ascii == "GET "
    assert context.packets[0].mitre_cosine_scores == {"T1001": 0.91}
    assert context.mitre_metadata["T1001"].tactic_id == "initial-access"


def test_subgraph_builder_falls_back_to_graph_store_after_hot_evict(tmp_path) -> None:
    mitre = _mitre_index()
    store = PersistentGraphStore(tmp_path / "graph_store", mitre_index=mitre, packet_embedding_dim=4)
    _append_packet(store)
    hot = HotGraphBuffer(mitre_index=mitre, ttl_seconds=0.1)
    hot.add_packet(
        packet_id="pkt_1",
        flow_id="flow_1",
        embedding=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        payload_hex="47455420",
        payload_ascii="GET ",
        payload_len_raw=4,
        timestamp=1.0,
        src_ip="192.168.1.10",
        dst_ip="10.0.0.5",
        src_port=51522,
        dst_port=80,
        protocol="TCP",
        mitre_topk=[("T1001", 0.91)],
    )
    hot.evict_expired(now=2.0)

    subgraph = SubgraphBuilder(
        hot,
        cold_store=store,
        protocol_mapping={"TCP": 1},
    ).build("flow_1")

    assert subgraph.node_features["flow"].shape == (1, 6)
    assert subgraph.node_features["packet"].shape == (1, 4)
    assert subgraph.node_features["tactic"].reshape(-1).tolist() == [0, 1]
    assert subgraph.packet_local_to_id == {0: "pkt_1"}


def test_context_hydrator_prefers_source_of_truth_store(tmp_path) -> None:
    mitre = _mitre_index()
    store = PersistentGraphStore(tmp_path / "graph_store", mitre_index=mitre, packet_embedding_dim=4)
    _append_packet(store, payload_ascii="STORE")

    hot = HotGraphBuffer(mitre_index=mitre)
    hot.add_packet(
        packet_id="pkt_1",
        flow_id="flow_1",
        embedding=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        payload_hex="484f5420",
        payload_ascii="HOT ",
        payload_len_raw=4,
        timestamp=1.0,
        src_ip="192.168.1.10",
        dst_ip="10.0.0.5",
        src_port=51522,
        dst_port=80,
        protocol="TCP",
        mitre_topk=[("T1001", 0.91)],
    )

    context = ContextHydrator().hydrate(
        "flow_1",
        hot_buffer=HotBufferAdapter(hot),
        cold_store=store,
    )

    assert context.packets[0].payload_preview_ascii == "STORE"


def test_training_loader_reads_sealed_graph_store_shards(tmp_path) -> None:
    root = tmp_path / "graph_store"
    store = PersistentGraphStore(root, mitre_index=_mitre_index(), packet_embedding_dim=4)
    _append_packet(store)
    store.seal_current_shard()

    artifact = load_graph_store_artifact(root, sealed_only=True)

    assert artifact.node_features["flow"].shape == (1, 6)
    assert artifact.node_features["packet"].shape == (1, 4)
    assert artifact.node_features["technique"].shape == (2, 4)
    assert artifact.flow_y.tolist() == [0]
    assert ("packet", "rev_matches_technique", "packet") not in artifact.edge_index
    assert ("packet", "rev_matches_technique", "flow") not in artifact.edge_index
    assert ("technique", "rev_matches_technique", "packet") in artifact.edge_index


def test_retention_drops_only_sealed_expired_shards(tmp_path) -> None:
    store = PersistentGraphStore(
        tmp_path / "graph_store",
        mitre_index=_mitre_index(),
        packet_embedding_dim=4,
        drop_after_days=0.001,
    )
    _append_packet(store)
    sealed = store.seal_current_shard()

    dropped = store.enforce_retention(now=200.0)

    assert dropped == [sealed]
    assert store.stats()["packets"] == 0
    assert store.load_context("flow_1") is None
