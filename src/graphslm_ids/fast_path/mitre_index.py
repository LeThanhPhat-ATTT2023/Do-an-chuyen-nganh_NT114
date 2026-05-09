from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class MitreIndex:
    """CPU cosine index for online MITRE technique lookup."""

    def __init__(
        self,
        embeddings_npy: str | Path,
        techniques_csv: str | Path,
        technique_tactic_csv: str | Path | None = None,
    ) -> None:
        embeddings = np.asarray(np.load(embeddings_npy), dtype=np.float32)
        techniques_df = pd.read_csv(techniques_csv)
        if "technique_id" not in techniques_df.columns:
            raise ValueError("techniques_csv must contain a technique_id column.")
        self._init_from_dataframes(embeddings, techniques_df, _read_optional_csv(technique_tactic_csv))

    @classmethod
    def from_arrays(
        cls,
        embeddings: np.ndarray,
        techniques: list[dict[str, Any]] | pd.DataFrame,
        technique_tactics: list[dict[str, Any]] | pd.DataFrame | None = None,
    ) -> "MitreIndex":
        obj = cls.__new__(cls)
        techniques_df = techniques if isinstance(techniques, pd.DataFrame) else pd.DataFrame(techniques)
        tactic_df = None
        if technique_tactics is not None:
            tactic_df = (
                technique_tactics
                if isinstance(technique_tactics, pd.DataFrame)
                else pd.DataFrame(technique_tactics)
            )
        obj._init_from_dataframes(np.asarray(embeddings, dtype=np.float32), techniques_df, tactic_df)
        return obj

    def topk(
        self,
        emb: np.ndarray,
        k: int = 5,
        threshold: float | None = None,
    ) -> list[tuple[str, float]]:
        query = np.asarray(emb, dtype=np.float32).reshape(-1)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        scores = self.embeddings @ query
        k = min(max(int(k), 0), int(scores.shape[0]))
        if k == 0:
            return []
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        pairs = sorted(((int(i), float(scores[int(i)])) for i in idx), key=lambda x: x[1], reverse=True)
        if threshold is not None:
            pairs = [item for item in pairs if item[1] >= float(threshold)]
        return [(self.technique_ids[i], score) for i, score in pairs]

    @property
    def technique_to_tactic(self) -> dict[str, str]:
        return dict(self._technique_to_tactic)

    @property
    def tactic_metadata(self) -> dict[str, dict[str, Any]]:
        return dict(self._tactic_metadata)

    @property
    def technique_metadata(self) -> dict[str, dict[str, Any]]:
        return dict(self._technique_metadata)

    def _init_from_dataframes(
        self,
        embeddings: np.ndarray,
        techniques_df: pd.DataFrame,
        technique_tactic_df: pd.DataFrame | None,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError("MITRE embeddings must be a 2D matrix.")
        if "technique_id" not in techniques_df.columns:
            raise ValueError("techniques data must contain technique_id.")
        if embeddings.shape[0] != techniques_df.shape[0]:
            raise ValueError("Technique embedding rows must align with techniques metadata.")

        self.embeddings = _l2_normalize(embeddings.astype(np.float32, copy=False))
        self.technique_ids = techniques_df["technique_id"].astype(str).tolist()
        self.technique_features = {
            tech_id: self.embeddings[idx].copy() for idx, tech_id in enumerate(self.technique_ids)
        }

        self._technique_to_tactic: dict[str, str] = {}
        self._tactic_metadata: dict[str, dict[str, Any]] = {}
        if technique_tactic_df is not None and not technique_tactic_df.empty:
            for row in technique_tactic_df.to_dict(orient="records"):
                tech_id = str(row.get("technique_id", ""))
                tactic_id = str(
                    row.get("tactic_id")
                    or row.get("tactic_shortname")
                    or row.get("tactic")
                    or row.get("tactic_name")
                    or ""
                )
                if tech_id and tactic_id and tech_id not in self._technique_to_tactic:
                    self._technique_to_tactic[tech_id] = tactic_id
                if tactic_id:
                    self._tactic_metadata.setdefault(
                        tactic_id,
                        {
                            "tactic_id": tactic_id,
                            "shortname": str(row.get("tactic_shortname") or tactic_id),
                            "name": str(row.get("tactic_name") or row.get("tactic") or tactic_id),
                        },
                    )

        self._technique_metadata: dict[str, dict[str, Any]] = {}
        for row in techniques_df.to_dict(orient="records"):
            tech_id = str(row.get("technique_id"))
            tactic_id = self._technique_to_tactic.get(tech_id, "")
            tactic_meta = self._tactic_metadata.get(tactic_id, {})
            self._technique_metadata[tech_id] = {
                "technique_id": tech_id,
                "technique_name": str(row.get("technique_name") or row.get("name") or tech_id),
                "name": str(row.get("technique_name") or row.get("name") or tech_id),
                "tactic": str(tactic_meta.get("shortname") or tactic_id or "unknown"),
                "tactic_id": tactic_id,
                "mapping_type": "embedding_cosine_similarity",
                "mapping_caution": (
                    "This link is based on semantic similarity, not deterministic signature matching."
                ),
            }


def _read_optional_csv(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)
