from __future__ import annotations

from pathlib import Path

import numpy as np


class StudentRuntime:
    """ONNX Runtime wrapper for the distilled student payload embedder."""

    def __init__(
        self,
        onnx_path: str | Path,
        providers: list[str] | None = None,
        normalize: bool = True,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for StudentRuntime. Install requirements-ml.txt."
            ) from exc

        self.onnx_path = Path(onnx_path)
        self.normalize = bool(normalize)
        session_options = ort.SessionOptions()
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=session_options,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed(self, payload_u8: np.ndarray) -> np.ndarray:
        batch = np.asarray(payload_u8, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch[None, :]
        output = self.embed_batch(batch)
        if output.shape[0] != 1:
            raise ValueError("embed expected a single payload row.")
        return output[0]

    def embed_batch(self, payload_batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(payload_batch)
        if batch.ndim != 2:
            raise ValueError("payload_batch must have shape (batch, payload_length).")
        batch = batch.astype(np.float32, copy=False)
        embeddings = self.session.run([self.output_name], {self.input_name: batch})[0]
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.normalize:
            embeddings = _l2_normalize(embeddings)
        return embeddings


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)
