from atlas.core import SemanticFieldLoader


def test_load_all_fields():
    atlas = SemanticFieldLoader("fields").load_all()

    expected = {
        "conceptual",
        "operational",
        "narrative",
        "scientific",
        "legal",
        "business",
    }

    assert expected.issubset(set(atlas.field_names()))

    for field in atlas.fields.values():
        assert field.embedding_dim == 1536
        assert field.rank > 0
        assert len(field.anchors) > 0