from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


Vector = np.ndarray
Matrix = np.ndarray


def normalize_vector(vector: Vector) -> Vector:
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def normalize_matrix(matrix: Matrix) -> Matrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_similarity(a: Vector, b: Vector) -> float:
    a = normalize_vector(a)
    b = normalize_vector(b)
    return float(np.dot(a, b))


@dataclass
class ContextMatrix:
    """
    Geometric representation of a semantic field.

    A ContextMatrix is built from anchor embeddings and defines
    a semantic subspace using SVD.

    Core criterion:
        O(z) = ||z - P_S(z)||²
    """

    name: str
    anchors: List[str]
    vectors: Matrix
    basis: Matrix
    singular_values: Vector
    centered: bool = False
    mean_vector: Vector | None = None

    @classmethod
    def from_vectors(
        cls,
        name: str,
        anchors: List[str],
        vectors: Matrix,
        centered: bool = False,
    ) -> "ContextMatrix":
        vectors = normalize_matrix(vectors)

        mean_vector = None
        matrix_for_svd = vectors

        if centered:
            mean_vector = np.mean(vectors, axis=0)
            matrix_for_svd = vectors - mean_vector

        context_matrix = matrix_for_svd.T

        U, singular_values, _ = np.linalg.svd(
            context_matrix,
            full_matrices=False,
        )

        basis = U[:, : len(anchors)]

        return cls(
            name=name,
            anchors=anchors,
            vectors=vectors,
            basis=basis,
            singular_values=singular_values,
            centered=centered,
            mean_vector=mean_vector,
        )

    def project(self, vector: Vector) -> Vector:
        z = normalize_vector(vector)

        if self.centered and self.mean_vector is not None:
            z = z - self.mean_vector

        projection = self.basis @ (self.basis.T @ z)

        if self.centered and self.mean_vector is not None:
            projection = projection + self.mean_vector

        return projection

    def origin_cost(self, vector: Vector) -> float:
        z = normalize_vector(vector)

        if self.centered and self.mean_vector is not None:
            z_for_projection = z - self.mean_vector
            projection = self.basis @ (self.basis.T @ z_for_projection)
            residual = z_for_projection - projection
        else:
            projection = self.basis @ (self.basis.T @ z)
            residual = z - projection

        return float(np.linalg.norm(residual) ** 2)

    def density_score(self, vector: Vector, top_k: int = 5) -> float:
        if len(self.vectors) == 0:
            return 0.0

        similarities = [
            cosine_similarity(vector, anchor_vector)
            for anchor_vector in self.vectors
        ]

        similarities = sorted(similarities, reverse=True)

        k = min(top_k, len(similarities))

        return float(sum(similarities[:k]) / k)

    def nearest_anchors(self, vector: Vector, top_k: int = 5) -> List[Tuple[str, float]]:
        scores = [
            (anchor, cosine_similarity(vector, anchor_vector))
            for anchor, anchor_vector in zip(self.anchors, self.vectors)
        ]

        scores.sort(key=lambda item: item[1], reverse=True)

        return scores[:top_k]

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    @property
    def embedding_dim(self) -> int:
        return int(self.basis.shape[0])