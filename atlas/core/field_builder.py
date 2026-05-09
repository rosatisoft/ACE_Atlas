import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List

import numpy as np

from .context_matrix import ContextMatrix, Matrix


EmbeddingFunction = Callable[[List[str]], Matrix]


@dataclass
class FieldBuildResult:
    name: str
    field_dir: Path
    anchor_count: int
    embedding_dim: int
    rank: int
    centered: bool


class SemanticFieldBuilder:
    """
    Builds and persists ACE Atlas semantic fields.

    This class does not know about OpenAI or any specific embedding provider.
    It receives an embedding function:

        embed_texts(List[str]) -> np.ndarray

    and turns anchors into:
        anchors.json
        vectors.npy
        basis.npy
        singular_values.npy
        metadata.json
    """

    def __init__(
        self,
        output_dir: str | Path,
        embed_texts: EmbeddingFunction,
        embedding_model: str,
        centered: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.embed_texts = embed_texts
        self.embedding_model = embedding_model
        self.centered = centered

    def build_field(
        self,
        name: str,
        anchors: List[str],
    ) -> FieldBuildResult:
        if not anchors:
            raise ValueError("anchors cannot be empty")

        field_dir = self.output_dir / name
        field_dir.mkdir(parents=True, exist_ok=True)

        vectors = self.embed_texts(anchors)

        matrix = ContextMatrix.from_vectors(
            name=name,
            anchors=anchors,
            vectors=vectors,
            centered=self.centered,
        )

        self._save_field(field_dir, matrix)

        return FieldBuildResult(
            name=name,
            field_dir=field_dir,
            anchor_count=len(anchors),
            embedding_dim=matrix.embedding_dim,
            rank=matrix.rank,
            centered=matrix.centered,
        )

    def _save_field(
        self,
        field_dir: Path,
        matrix: ContextMatrix,
    ) -> None:
        anchors_payload = {
            "field": matrix.name,
            "anchors": matrix.anchors,
        }

        metadata = {
            "field": matrix.name,
            "embedding_model": self.embedding_model,
            "method": "svd_context_matrix",
            "normalization": True,
            "centered": matrix.centered,
            "anchor_count": len(matrix.anchors),
            "embedding_dim": matrix.embedding_dim,
            "rank": matrix.rank,
            "basis_shape": list(matrix.basis.shape),
            "vectors_shape": list(matrix.vectors.shape),
            "singular_values_top10": matrix.singular_values[:10].tolist(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        (field_dir / "anchors.json").write_text(
            json.dumps(anchors_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        (field_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        np.save(field_dir / "vectors.npy", matrix.vectors)
        np.save(field_dir / "basis.npy", matrix.basis)
        np.save(field_dir / "singular_values.npy", matrix.singular_values)

        if matrix.mean_vector is not None:
            np.save(field_dir / "mean_vector.npy", matrix.mean_vector)