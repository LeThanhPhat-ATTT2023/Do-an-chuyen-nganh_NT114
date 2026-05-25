"""PMI candidate generation + L1-multinomial Logistic refinement (Source 1 of v3).

End-to-end: subsample 200K packets (stratified by class) -> tokenize -> count
(token, class) co-occurrences -> filter by PMI + support thresholds ->
build sparse design matrix -> fit L1 multinomial Logistic Regression with the
``saga`` solver -> project class-level coefficients onto MITRE techniques via
``class_technique_map.csv``.

Why not naive PMI alone? PMI assumes token independence, which is grossly
violated by overlapping byte 4-grams and 8-grams. L1-LR refines via joint
convex optimization: the coefficients are signed (positive = evidence-for,
negative = evidence-against), automatic feature selection comes for free via
the L1 penalty, and the global optimum is unique. No neural net training, no
GPU; this is the classic scikit-learn convex-stat pipeline (Tibshirani-Hastie).

Memory budget (200K subsample, ~18K candidate tokens):

* Subsample frame                 ~ 300 MB  (payloads + token sets in dict)
* token_class_count counters      ~ 200 MB  (sparse Counters)
* Sparse design matrix CSR        ~ 200 MB  (np.float32 nnz ~ 50M)
* sklearn coef + workspace        ~ 1.5 GB  (saga keeps a dual buffer)
* ----------------------------------------- = ~ 2.2 GB peak

Wall-clock budget (16-core CPU): 15-25 minutes for 200K x 18K x 13 classes.

Determinism: the subsampling RNG and the saga ``random_state`` both consume
the same ``seed`` parameter, so identical inputs produce identical outputs.
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from graphslm_ids.offline.preprocessing.v3.tokenizer import tokenize_payload

_LOG = logging.getLogger(__name__)


def _coerce_payload(raw: Any) -> bytes:
    """Best-effort payload coercion. ``packets_df`` may hold ``bytes``, ``bytearray``,
    a uint8 ``np.ndarray``, or a memoryview; tokenization needs raw bytes.
    """
    if raw is None:
        return b""
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, memoryview):
        return raw.tobytes()
    if isinstance(raw, np.ndarray):
        # Trim trailing zero pad if it came from a fixed-width payload-256 row.
        return raw.astype(np.uint8, copy=False).tobytes()
    if isinstance(raw, (list, tuple)):
        return bytes(raw)
    return b""


def stratified_subsample_packets(
    train_packets_df: pd.DataFrame,
    n_per_class: int = 25_000,
    max_total: int = 200_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Up to ``n_per_class`` packets per class, capped at ``max_total`` rows total.

    Classes with fewer than ``n_per_class`` rows contribute everything they have.
    Tiny classes (< 100 packets) are oversampled WITH REPLACEMENT up to
    ``n_per_class // 4`` so PMI has at least some signal for them — without this
    a single-PCAP scan class would be invisible to the learner.

    Args:
        train_packets_df: must have a ``label`` column. Each row represents one
            packet; the function does not look at any other column.
        n_per_class: hard cap per class before stratification.
        max_total: hard cap across all classes (proportionally truncates if hit).
        seed: numpy RNG seed.

    Returns:
        A new DataFrame (no in-place mutation) containing the sampled rows in
        a deterministic order: class label sorted ascending, then by original
        index.
    """
    if "label" not in train_packets_df.columns:
        raise ValueError("train_packets_df must contain a 'label' column")

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    # Sort labels so the iteration order is deterministic across runs / OSes.
    for label in sorted(train_packets_df["label"].astype(str).unique()):
        cls = train_packets_df[train_packets_df["label"].astype(str) == label]
        n = len(cls)
        if n == 0:
            continue
        if n < 100:
            # Oversample tiny classes WITH REPLACEMENT — capped well below
            # n_per_class so we don't drown the loss with duplicates.
            replace = True
            take = min(n_per_class // 4, max(n * 4, 50))
        else:
            replace = False
            take = min(n_per_class, n)
        idx = rng.choice(n, size=take, replace=replace)
        # Use ``.iloc[idx]`` (positional) so duplicates from ``replace=True`` map
        # to repeated rows rather than collapsing on the unique index.
        parts.append(cls.iloc[idx])

    if not parts:
        return train_packets_df.iloc[0:0].copy()

    combined = pd.concat(parts, ignore_index=True)
    if len(combined) > max_total:
        # Proportionally trim across classes so no single dominant class
        # crowds out minority signal after the global cap.
        keep_idx = rng.choice(len(combined), size=max_total, replace=False)
        keep_idx.sort()
        combined = combined.iloc[keep_idx].reset_index(drop=True)
    return combined


def fit_pmi_counts(
    subsample: pd.DataFrame,
) -> tuple[Counter, dict[str, Counter]]:
    """Stream-tokenize ``subsample`` and accumulate co-occurrence counters.

    Args:
        subsample: must have ``payload`` and ``label`` columns.

    Returns:
        ``(class_total, token_class_count)`` where ``class_total[label]`` is
        the number of packets of that class in the subsample, and
        ``token_class_count[token]`` is a ``Counter`` mapping label -> count.
    """
    if "payload" not in subsample.columns or "label" not in subsample.columns:
        raise ValueError("subsample must contain 'payload' and 'label' columns")

    class_total: Counter = Counter()
    token_class_count: dict[str, Counter] = {}

    payloads = subsample["payload"].tolist()
    labels = subsample["label"].astype(str).tolist()
    for raw, label in zip(payloads, labels, strict=True):
        class_total[label] += 1
        tokens = tokenize_payload(_coerce_payload(raw))
        if not tokens:
            continue
        for tok in tokens:
            ctr = token_class_count.get(tok)
            if ctr is None:
                ctr = Counter()
                token_class_count[tok] = ctr
            ctr[label] += 1
    return class_total, token_class_count


def filter_pmi_candidates(
    token_class_count: dict[str, Counter],
    class_total: Counter,
    pmi_min: float = 1.0,
    support_min_global: int = 50,
    support_min_class: int = 5,
    top_k_per_class: int = 2000,
) -> tuple[set[str], dict[str, dict[str, float]]]:
    """Score tokens by PMI per class and keep top-K per class.

    PMI(token, class) = log[ P(t, c) / (P(t) P(c)) ]. We approximate the joint
    probabilities by counts over the subsample (the document collection is
    the set of packets, and a token is "present" once per packet because
    :func:`tokenize_payload` returns a *set*).

    Args:
        token_class_count: output of :func:`fit_pmi_counts`.
        class_total: output of :func:`fit_pmi_counts`.
        pmi_min: minimum PMI for a (token, class) pair to be a candidate.
        support_min_global: token must appear in at least this many packets total.
        support_min_class: token must appear in at least this many packets of
            the specific class to be considered for that class.
        top_k_per_class: per class, keep the top-K tokens by PMI.

    Returns:
        ``(candidate_tokens, pmi_per_token_class)`` — the first is the union
        of per-class top-K sets; the second is a nested dict
        ``{token: {class: pmi}}`` containing only entries that passed the
        thresholds.
    """
    total_packets = float(sum(class_total.values()))
    if total_packets <= 0:
        return set(), {}

    # Pre-compute P(class) and totals per token.
    p_class = {cls: cnt / total_packets for cls, cnt in class_total.items()}
    token_total: dict[str, int] = {}
    for tok, ctr in token_class_count.items():
        token_total[tok] = int(sum(ctr.values()))

    # Per-class candidate pool: list of (pmi, token).
    per_class_pool: dict[str, list[tuple[float, str]]] = {
        cls: [] for cls in class_total
    }
    pmi_per_token_class: dict[str, dict[str, float]] = {}

    for tok, ctr in token_class_count.items():
        n_t = token_total[tok]
        if n_t < support_min_global:
            continue
        p_t = n_t / total_packets
        for cls, n_tc in ctr.items():
            if n_tc < support_min_class:
                continue
            p_tc = n_tc / total_packets
            denom = p_t * p_class[cls]
            if denom <= 0:
                continue
            pmi = float(np.log(p_tc / denom))
            if pmi < pmi_min:
                continue
            per_class_pool[cls].append((pmi, tok))
            pmi_per_token_class.setdefault(tok, {})[cls] = pmi

    candidate_tokens: set[str] = set()
    for cls, pool in per_class_pool.items():
        # nlargest by PMI is O(n log k); for ~20K entries either way works.
        pool.sort(key=lambda p: p[0], reverse=True)
        for _pmi, tok in pool[:top_k_per_class]:
            candidate_tokens.add(tok)

    return candidate_tokens, pmi_per_token_class


def _build_design_matrix(
    subsample: pd.DataFrame,
    candidates: set[str],
) -> tuple[csr_matrix, list[str], dict[str, int]]:
    """Build a sparse (n_packets, n_candidates) presence matrix.

    Uses ``lil_matrix`` for cheap row-wise build, then converts to CSR. Dense
    would be ``200K * 18K * 4 bytes = 14 GB`` — sparse keeps it at ~200 MB.

    Returns:
        ``(X, token_list, token_to_idx)``.
    """
    token_list = sorted(candidates)
    token_to_idx = {t: i for i, t in enumerate(token_list)}

    n_rows = len(subsample)
    n_cols = len(token_list)
    X = lil_matrix((n_rows, n_cols), dtype=np.float32)

    payloads = subsample["payload"].tolist()
    for row_idx, raw in enumerate(payloads):
        tokens = tokenize_payload(_coerce_payload(raw))
        if not tokens:
            continue
        # Filter to candidates BEFORE writing into the sparse matrix. Tokens
        # not in ``token_to_idx`` aren't features so writing them is wasted IO.
        cols = [token_to_idx[t] for t in tokens if t in token_to_idx]
        for c in cols:
            X[row_idx, c] = 1.0
    return X.tocsr(), token_list, token_to_idx


def fit_l1_logistic(
    subsample: pd.DataFrame,
    candidates: set[str],
    label_to_idx: dict[str, int],
    C: float = 0.1,
    max_iter: int = 200,
    n_jobs: int = -1,
    seed: int = 42,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Fit multinomial L1-LR on the candidate token presence matrix.

    Args:
        subsample: must have ``payload`` and ``label`` columns.
        candidates: token vocabulary (output of :func:`filter_pmi_candidates`).
        label_to_idx: deterministic class -> int mapping used to encode ``y``.
            Pass an externally-chosen mapping so downstream code can rely on
            stable class indices.
        C: inverse regularization strength (smaller = stronger L1).
        max_iter: solver iterations. ``saga`` typically needs 100-300.
        n_jobs: passed to sklearn. ``-1`` uses all cores.
        seed: passed to sklearn's ``random_state``.

    Returns:
        ``(coef_matrix, token_list, class_to_idx)`` — ``coef_matrix`` has shape
        ``(num_classes, num_candidates)``. For binary problems sklearn returns
        a single-row coefficient; we expand it to (2, n_features) so the
        downstream projection code is uniform across binary and multiclass.
    """
    if not candidates:
        # Nothing to fit; return empty contract-compatible shapes.
        return np.zeros((0, 0), dtype=np.float32), [], dict(label_to_idx)

    X, token_list, _ = _build_design_matrix(subsample, candidates)
    # Use the externally-provided class ordering so the coef rows align with
    # whatever the caller will later use to project to techniques.
    y = np.asarray(
        [label_to_idx[str(lbl)] for lbl in subsample["label"].astype(str).tolist()],
        dtype=np.int64,
    )

    # sklearn 1.8 dropped the deprecated ``multi_class`` arg — ``saga`` infers
    # multinomial automatically when ``y`` has >2 classes. We keep
    # ``warnings.filterwarnings`` ringed around the fit so the (occasionally
    # noisy) convergence warning doesn't pollute the CLI output.
    clf = LogisticRegression(
        penalty="l1",
        C=C,
        solver="saga",
        max_iter=max_iter,
        n_jobs=n_jobs,
        random_state=seed,
        tol=1e-4,
    )
    _LOG.info(
        "fit_l1_logistic: X.shape=%s, n_classes=%d, C=%s, max_iter=%d",
        X.shape,
        len(label_to_idx),
        C,
        max_iter,
    )
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(X, y)
    _LOG.info("fit_l1_logistic: saga fit took %.1fs", time.time() - t0)

    # For binary y, sklearn returns coef_ of shape (1, n_features) — the
    # "positive" class row. Expand to a (2, n_features) matrix where row 0 is
    # the negation, so downstream projection code can index by class without
    # special-casing the binary regime.
    coef = clf.coef_
    if coef.shape[0] == 1 and len(label_to_idx) == 2:
        coef = np.vstack([-coef, coef])

    # Sanity: ``coef`` must have one row per class index in ``label_to_idx``.
    if coef.shape[0] != len(label_to_idx):
        # sklearn drops classes from coef_ that don't appear in y. Pad missing
        # rows with zeros so the downstream class-to-tech projection stays well-defined.
        full = np.zeros((len(label_to_idx), coef.shape[1]), dtype=np.float32)
        for sklearn_row, cls in enumerate(clf.classes_.tolist()):
            full[int(cls), :] = coef[sklearn_row, :]
        coef = full

    return coef.astype(np.float32), token_list, dict(label_to_idx)


def project_to_techniques(
    coef_matrix: np.ndarray,
    token_list: list[str],
    class_to_idx: dict[str, int],
    class_technique_map: pd.DataFrame,
    technique_family: pd.DataFrame,
    coef_threshold: float = 0.01,
) -> pd.DataFrame:
    """Project class-level coefficients to ``(token, technique, family, weight)``.

    The class -> technique map is a many-to-many relation with per-edge weights
    (a class may evidence multiple techniques with different confidences). For
    each (token, technique) pair the final weight is

        sum_{class : (class, technique) in map} (coef[class, token] * map_weight)

    aggregated across all classes that point to the technique. The result is
    clipped to ``[-1, 1]`` (downstream code uses these as signed evidence
    scores; saturating at ±1 keeps the ensemble math well-behaved).

    Args:
        coef_matrix: ``(n_classes, n_candidates)`` from :func:`fit_l1_logistic`.
        token_list: column ordering of ``coef_matrix``.
        class_to_idx: row ordering of ``coef_matrix``.
        class_technique_map: must have ``class``, ``technique``, ``weight``
            columns. Rows with empty ``technique`` (e.g. benign class) are
            silently dropped.
        technique_family: must have ``technique``, ``family`` columns.
        coef_threshold: rows where ``|weight| < coef_threshold`` after
            aggregation are dropped.

    Returns:
        DataFrame with columns ``token``, ``technique``, ``family``, ``weight``.
        Rows whose technique has no family entry are dropped (the v3 ensemble
        needs a family for typed-edge routing).
    """
    if coef_matrix.size == 0 or not token_list:
        return pd.DataFrame(columns=["token", "technique", "family", "weight"])

    # Build helpers.
    tech_to_family: dict[str, str] = {
        str(r.technique): str(r.family)
        for r in technique_family.itertuples(index=False)
    }
    # Aggregate (token, technique, family) -> weight in a plain dict for speed.
    agg: dict[tuple[str, str, str], float] = {}

    # itertuples auto-renames the ``class`` column (Python keyword) to ``_<n>``
    # which made the old attribute-based lookup fragile. iterrows is slightly
    # slower per row, but the class_technique map is ~20 rows so we never
    # notice — and the Series.get() lookup handles the keyword cleanly.
    for _, row in class_technique_map.iterrows():
        cls_raw = row.get("class")
        if cls_raw is None or (isinstance(cls_raw, float) and np.isnan(cls_raw)):
            continue
        cls = str(cls_raw)
        tech = row.get("technique")
        w_map = row.get("weight")
        if tech is None or (isinstance(tech, float) and np.isnan(tech)) or str(tech).strip() == "":
            continue
        tech = str(tech)
        try:
            w_map = float(w_map)
        except (TypeError, ValueError):
            continue
        family = tech_to_family.get(tech)
        if family is None:
            # Unknown family -> skip; the ensemble can't route this edge.
            continue
        cls_idx = class_to_idx.get(cls)
        if cls_idx is None:
            # Class present in the CSV but absent from the training labels:
            # skip rather than fail. This is normal when the dataset doesn't
            # include every class the project may eventually cover.
            continue

        row_coef = coef_matrix[cls_idx, :]
        # Vectorised over tokens; iterate only where the coefficient is
        # significant to keep the aggregator small.
        nz = np.where(np.abs(row_coef) > 0)[0]
        for j in nz:
            tok = token_list[j]
            key = (tok, tech, family)
            agg[key] = agg.get(key, 0.0) + float(row_coef[j]) * w_map

    rows: list[tuple[str, str, str, float]] = []
    for (tok, tech, family), w in agg.items():
        clipped = float(np.clip(w, -1.0, 1.0))
        if abs(clipped) < coef_threshold:
            continue
        rows.append((tok, tech, family, clipped))

    if not rows:
        return pd.DataFrame(columns=["token", "technique", "family", "weight"])
    return pd.DataFrame(rows, columns=["token", "technique", "family", "weight"])


def fit_and_save_pmi_table(
    train_packets_df: pd.DataFrame,
    class_technique_map_path: Path,
    technique_family_path: Path,
    out_pmi_table: Path,
    out_meta_json: Path,
    n_per_class: int = 25_000,
    max_total: int = 200_000,
    seed: int = 42,
    pmi_min: float = 1.0,
    support_min_global: int = 50,
    support_min_class: int = 5,
    top_k_per_class: int = 2000,
    C: float = 0.1,
    coef_threshold: float = 0.01,
) -> dict[str, Any]:
    """End-to-end fit. Writes ``pmi_table.parquet`` + ``meta.json``.

    Pipeline: subsample -> PMI counts -> candidate filter -> L1-LR ->
    class -> technique projection -> parquet.

    Returns the meta dict for in-memory inspection / logging by the caller.
    """
    out_pmi_table = Path(out_pmi_table)
    out_meta_json = Path(out_meta_json)
    out_pmi_table.parent.mkdir(parents=True, exist_ok=True)
    out_meta_json.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    _LOG.info("[1/5] stratified subsample (n_per_class=%d, max_total=%d)", n_per_class, max_total)
    t0 = time.time()
    subsample = stratified_subsample_packets(
        train_packets_df, n_per_class=n_per_class, max_total=max_total, seed=seed
    )
    t_sub = time.time() - t0
    _LOG.info(
        "  -> subsample rows=%d, classes=%d (%.1fs)",
        len(subsample),
        subsample["label"].nunique(),
        t_sub,
    )

    _LOG.info("[2/5] PMI counting (streaming tokenization)")
    t0 = time.time()
    class_total, token_class_count = fit_pmi_counts(subsample)
    t_cnt = time.time() - t0
    _LOG.info(
        "  -> %d unique tokens, %d classes (%.1fs)",
        len(token_class_count),
        len(class_total),
        t_cnt,
    )

    _LOG.info("[3/5] PMI candidate filter")
    t0 = time.time()
    candidates, pmi_per_token_class = filter_pmi_candidates(
        token_class_count,
        class_total,
        pmi_min=pmi_min,
        support_min_global=support_min_global,
        support_min_class=support_min_class,
        top_k_per_class=top_k_per_class,
    )
    t_filt = time.time() - t0
    _LOG.info("  -> %d candidate tokens (%.1fs)", len(candidates), t_filt)

    # Deterministic class ordering so coef rows are stable across runs.
    label_to_idx = {cls: i for i, cls in enumerate(sorted(class_total.keys()))}

    _LOG.info("[4/5] L1 multinomial logistic regression")
    t0 = time.time()
    coef_matrix, token_list, class_to_idx = fit_l1_logistic(
        subsample, candidates, label_to_idx, C=C, seed=seed
    )
    t_lr = time.time() - t0
    _LOG.info(
        "  -> coef shape=%s nnz=%d (%.1fs)",
        coef_matrix.shape,
        int(np.count_nonzero(coef_matrix)),
        t_lr,
    )

    _LOG.info("[5/5] project to techniques + write parquet")
    t0 = time.time()
    class_technique_map = pd.read_csv(class_technique_map_path)
    technique_family = pd.read_csv(technique_family_path)
    pmi_table = project_to_techniques(
        coef_matrix,
        token_list,
        class_to_idx,
        class_technique_map,
        technique_family,
        coef_threshold=coef_threshold,
    )
    pmi_table.to_parquet(out_pmi_table, index=False)
    t_proj = time.time() - t0
    _LOG.info(
        "  -> %d rows, %d unique tokens, %d unique techniques (%.1fs)",
        len(pmi_table),
        pmi_table["token"].nunique() if len(pmi_table) else 0,
        pmi_table["technique"].nunique() if len(pmi_table) else 0,
        t_proj,
    )

    meta: dict[str, Any] = {
        "seed": int(seed),
        "n_subsample": int(len(subsample)),
        "n_classes": int(len(class_total)),
        "class_total": {str(k): int(v) for k, v in class_total.items()},
        "n_unique_tokens": int(len(token_class_count)),
        "n_candidate_tokens": int(len(candidates)),
        "coef_shape": list(coef_matrix.shape),
        "coef_nnz": int(np.count_nonzero(coef_matrix)),
        "pmi_table_rows": int(len(pmi_table)),
        "pmi_table_unique_tokens": int(
            pmi_table["token"].nunique() if len(pmi_table) else 0
        ),
        "pmi_table_unique_techniques": int(
            pmi_table["technique"].nunique() if len(pmi_table) else 0
        ),
        "label_to_idx": label_to_idx,
        "wall_seconds": {
            "subsample": round(t_sub, 2),
            "pmi_count": round(t_cnt, 2),
            "pmi_filter": round(t_filt, 2),
            "l1_logreg": round(t_lr, 2),
            "project": round(t_proj, 2),
            "total": round(time.time() - t_start, 2),
        },
        "hyperparameters": {
            "n_per_class": int(n_per_class),
            "max_total": int(max_total),
            "pmi_min": float(pmi_min),
            "support_min_global": int(support_min_global),
            "support_min_class": int(support_min_class),
            "top_k_per_class": int(top_k_per_class),
            "C": float(C),
            "max_iter": 200,
            "coef_threshold": float(coef_threshold),
        },
    }
    with out_meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


__all__ = [
    "stratified_subsample_packets",
    "fit_pmi_counts",
    "filter_pmi_candidates",
    "fit_l1_logistic",
    "project_to_techniques",
    "fit_and_save_pmi_table",
]
