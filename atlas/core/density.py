from typing import Dict

import numpy as np

from .context_matrix import ContextMatrix, cosine_similarity


def field_density(
    field: ContextMatrix,
    vector: np.ndarray,
    top_k: int = 5,
) -> float:
    """
    Estimate semantic density inside a field.

    Density is computed as the average cosine similarity
    against the nearest anchor vectors.
    """

    if len(field.vectors) == 0:
        return 0.0

    similarities = [
        cosine_similarity(vector, anchor_vector)
        for anchor_vector in field.vectors
    ]

    similarities = sorted(similarities, reverse=True)

    k = min(top_k, len(similarities))

    return float(sum(similarities[:k]) / k)


def atlas_density(
    fields: Dict[str, ContextMatrix],
    vector: np.ndarray,
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Compute density scores across all atlas fields.
    """

    return {
        field_name: field_density(field, vector, top_k=top_k)
        for field_name, field in fields.items()
    }