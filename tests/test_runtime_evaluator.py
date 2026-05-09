import numpy as np

from atlas.core import AtlasRuntimeEvaluator, SemanticFieldLoader


def test_runtime_evaluator_returns_valid_result():
    atlas = SemanticFieldLoader("fields").load_all()
    evaluator = AtlasRuntimeEvaluator(atlas)

    field = atlas.get_field("legal")
    vector = field.vectors[0]

    result = evaluator.evaluate(
        text="legal anchor",
        vector=vector,
    )

    assert result.best_field in atlas.field_names()
    assert result.best_cost >= 0
    assert result.field_margin >= 0
    assert result.stability_index >= 0
    assert result.interpretation