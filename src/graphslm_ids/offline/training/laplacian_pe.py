"""HPE-HGT Laplacian Positional Encoding.

Reference:
  - Dwivedi et al. "Graph Neural Networks with Learnable Structural and
    Positional Representations" (ICLR 2022) — Laplacian PE definition.
  - MTSU 2025 HPE-HGT for IDS (spec §3.3) — application to heterogeneous
    Graph Transformer for intrusion detection.

Key idea: precompute the k smallest non-trivial eigenvectors of the graph
Laplacian once, then concatenate them as additional node features. This
gives the HGT global topology awareness it otherwise lacks (HGT's
attention is purely local relation-aware).

For NT114's three-tier graph, this module operates on the FLOW node
sub-relation graph derived from packet co-flow co-occurrence. Run once
offline; cache to disk; sampler looks up rows for each subgraph.

Public API:
  - :func:`compute_laplacian_pe` — top-k eigenvectors of symmetric Laplacian.
  - :func:`save_laplacian_pe` / :func:`load_laplacian_pe` — cache helpers.
  - :func:`concat_pe_to_features` — concat PE rows to node features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_laplacian_pe(
    edge_index: np.ndarray,
    num_nodes: int,
    k: int = 16,
    seed: int | None = None,
) -> np.ndarray:
    """Compute top-k non-trivial Laplacian eigenvectors for positional encoding.

    Builds the symmetric normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}
    where A is symmetrized from edge_index (we add reverse edges and remove
    self-loops automatically). Then returns the smallest-k eigenvectors as
    a ``(num_nodes, k)`` matrix.

    Sign-flipping: Laplacian eigenvectors have arbitrary sign. We fix the
    sign so each eigenvector's first non-zero entry is positive —
    making the output deterministic given seed.

    Args:
        edge_index: ``(2, E)`` array of (src, dst) pairs. Direction is
            ignored — we symmetrize internally.
        num_nodes: Total number of nodes in the graph.
        k: Number of eigenvectors to return. If ``k > num_nodes - 1``,
            extra columns are zero-padded.
        seed: Optional seed for Lanczos starting vector (only relevant
            for sparse iterative solver path).

    Returns:
        ``(num_nodes, k)`` float32 array.
    """
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be > 0, got {num_nodes}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # Symmetrize edge_index, drop self-loops + duplicates.
    src, dst = edge_index[0].astype(np.int64), edge_index[1].astype(np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])
    pair_view = np.stack([all_src, all_dst], axis=1)
    # Deduplicate pairs.
    pair_view = np.unique(pair_view, axis=0)
    sym_src = pair_view[:, 0]
    sym_dst = pair_view[:, 1]

    # Build sparse adjacency, then degree vector.
    from scipy.sparse import csr_matrix, eye
    data = np.ones(sym_src.shape[0], dtype=np.float64)
    A = csr_matrix((data, (sym_src, sym_dst)), shape=(num_nodes, num_nodes))
    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    # D^{-1/2} with safe zero-handling for isolated nodes.
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    D_inv_sqrt = csr_matrix(
        (deg_inv_sqrt, (np.arange(num_nodes), np.arange(num_nodes))),
        shape=(num_nodes, num_nodes),
    )
    # L_sym = I - D^{-1/2} A D^{-1/2}
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    L = eye(num_nodes, format="csr") - A_norm

    # Solve for smallest k+1 eigenpairs, drop the trivial 0 eigenvector at idx 0.
    k_eff = min(k + 1, num_nodes)
    pe_full = np.zeros((num_nodes, k), dtype=np.float32)
    if num_nodes == 1:
        return pe_full
    if k_eff < 2:
        return pe_full

    if num_nodes <= 50:
        # Dense path is more reliable for small graphs; iterative solvers
        # need at least n > 2k for convergence.
        L_dense = L.toarray()
        eigvals, eigvecs = np.linalg.eigh(L_dense)
        # Sort ascending (eigh returns in ascending order, but be explicit).
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
    else:
        from scipy.sparse.linalg import eigsh
        rng = np.random.default_rng(seed if seed is not None else 0)
        v0 = rng.normal(size=num_nodes).astype(np.float64)
        try:
            eigvals, eigvecs = eigsh(L, k=k_eff, which="SM", v0=v0, tol=1e-6)
        except Exception:
            # Fallback to dense if Lanczos fails (small or degenerate graphs).
            L_dense = L.toarray()
            eigvals, eigvecs = np.linalg.eigh(L_dense)
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

    # Drop trivial zero eigenvector (idx 0). Take next k.
    n_take = min(k, eigvecs.shape[1] - 1)
    pe = eigvecs[:, 1 : 1 + n_take].astype(np.float32)

    # Deterministic sign: flip each column so first non-zero entry is positive.
    for col in range(pe.shape[1]):
        first_nz_idx = np.argmax(np.abs(pe[:, col]) > 1e-8)
        if pe[first_nz_idx, col] < 0:
            pe[:, col] = -pe[:, col]

    pe_full[:, :n_take] = pe
    return pe_full


def save_laplacian_pe(pe: np.ndarray, path: str | Path) -> None:
    """Save PE array to a .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, pe.astype(np.float32))


def load_laplacian_pe(path: str | Path) -> np.ndarray:
    """Load PE array from .npy; coerces to float32."""
    pe = np.load(Path(path))
    return pe.astype(np.float32)


def concat_pe_to_features(
    features: np.ndarray,
    pe: np.ndarray,
    global_indices: np.ndarray | None = None,
    pad_to_dim: int | None = None,
) -> np.ndarray:
    """Concatenate Laplacian PE columns onto node features.

    Args:
        features: ``(N, D_feat)`` array.
        pe: Either ``(N, D_pe)`` aligned to features, OR a full ``(N_total, D_pe)``
            array if ``global_indices`` is provided.
        global_indices: When the subgraph contains a subset of the full
            graph's nodes, pass the global IDs (length N) and ``pe`` is
            the full-graph PE; rows are gathered.
        pad_to_dim: If provided, zero-pad PE columns up to this width.

    Returns:
        ``(N, D_feat + D_pe_effective)`` float32 array.
    """
    if global_indices is not None:
        pe_rows = pe[np.asarray(global_indices, dtype=np.int64)]
    else:
        pe_rows = pe
        if pe_rows.shape[0] != features.shape[0]:
            raise ValueError(
                f"pe rows {pe_rows.shape[0]} != features rows {features.shape[0]} "
                "(pass global_indices for subgraph case)"
            )

    if pad_to_dim is not None:
        cur_dim = pe_rows.shape[1]
        if cur_dim < pad_to_dim:
            pad = np.zeros((pe_rows.shape[0], pad_to_dim - cur_dim), dtype=pe_rows.dtype)
            pe_rows = np.concatenate([pe_rows, pad], axis=1)
        elif cur_dim > pad_to_dim:
            pe_rows = pe_rows[:, :pad_to_dim]

    out = np.concatenate(
        [features.astype(np.float32, copy=False), pe_rows.astype(np.float32, copy=False)],
        axis=1,
    )
    return out
