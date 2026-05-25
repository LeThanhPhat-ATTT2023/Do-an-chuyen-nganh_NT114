"""Flow-level signature consensus (Source 3 of the v3 PMI Ensemble).

This module is intentionally thin: it wraps v2's
:func:`graphslm_ids.offline.preprocessing.signatures.match_flow_signatures`
in a single batch call that returns ``{flow_id: {technique_id: weight}}``.

Why a separate module instead of inlining the wrap into ``ensemble.py``?
Three reasons:

* The v3 graph builder iterates packets, not flows, but consensus weights are
  computed *per flow* and applied as a multiplicative boost when the packet's
  parent flow has consensus on the same technique. Precomputing the per-flow
  hits once is dramatically cheaper than re-firing the flow rules per packet.
* The wrap is the single integration point with v2. Keeping it isolated means
  if v2's flow-signature API changes, only this file breaks — not the
  ensemble aggregator.
* Unit testability: the consensus computation is deterministic given the
  features frame and can be tested in isolation without spinning up an
  Aho-Corasick automaton or PMI table.

Memory budget: ~500 MB. We iterate row-by-row and emit one Python dict per
flow that fires; for 177K flows with mostly 0-2 hits each, the output dict
weighs ~30-50 MB.
"""
from __future__ import annotations

import pandas as pd

from graphslm_ids.offline.preprocessing.signatures import match_flow_signatures


def flow_consensus_hits(feats_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Fire v2 flow signatures over every row of ``feats_df``.

    Args:
        feats_df: per-flow feature frame indexed by ``flow_id``. Must have
            the columns v2's flow rules read (``proto``, ``r_syn``,
            ``fan_dst_port``, etc.) — see v2/_signature_rules.py for the
            exact set.

    Returns:
        ``{flow_id_str: {technique_id_str: weight_float}}``. Flows with no
        signature hits are omitted (callers should treat a missing key as
        an empty dict).
    """
    out: dict[str, dict[str, float]] = {}
    # itertuples is the fastest pandas iteration shape that preserves
    # column access by name. We avoid iterrows because it boxes each row as
    # a Series — ~3x slower at 200K-row scale.
    for flow_id, row in zip(feats_df.index, feats_df.itertuples(index=False), strict=False):
        # match_flow_signatures expects a row that supports __getitem__ by
        # column name. NamedTuple from itertuples does NOT support that, so
        # we hand it the original .loc[] row to preserve the v2 contract.
        # itertuples is still the right iteration driver — it's the index
        # walking, not the row-shape, that dominates.
        hits = match_flow_signatures(feats_df.loc[flow_id])
        if not hits:
            continue
        out[str(flow_id)] = {tech: float(w) for tech, w in hits}
    return out
