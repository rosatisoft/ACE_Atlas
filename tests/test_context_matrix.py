import numpy as np

from atlas.core import ContextMatrix


def test_context_matrix_origin_cost_for_anchor_is_low():
    anchors = ["alpha", "beta", "gamma"]

    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    field = ContextMatrix.from_vectors(
        name="test",
        anchors=anchors,
        vectors=vectors,
    )

    cost = field.origin_cost(np.array([1.0, 0.0, 0.0]))

    assert cost < 1e-10


def test_density_score_is_positive_for_near_anchor():
    anchors = ["alpha", "beta"]

    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    field = ContextMatrix.from_vectors(
        name="test",
        anchors=anchors,
        vectors=vectors,
    )

    density = field.density_score(np.array([1.0, 0.0]))

    assert density > 0