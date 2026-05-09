import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .context_matrix import ContextMatrix


@dataclass
class LoadedAtlas:
    fields: Dict[str, ContextMatrix]
    metadata: Dict[str, dict]

    def field_names(self) -> List[str]:
        return list(self.fields.keys())

    def get_field(self, name: str) -> ContextMatrix:
        if name not in self.fields:
            raise KeyError(f"Unknown field: {name}")

        return self.fields[name]


class SemanticFieldLoader:
    """
    Loads persisted ACE Atlas semantic fields from disk.

    Expected field structure:

        fields/
          conceptual/
            anchors.json
            vectors.npy
            basis.npy
            singular_values.npy
            metadata.json
          operational/
            ...
    """

    def __init__(self, fields_dir: str | Path) -> None:
        self.fields_dir = Path(fields_dir)

    def load_field(self, name: str) -> ContextMatrix:
        field_dir = self.fields_dir / name

        if not field_dir.exists():
            raise FileNotFoundError(f"Field directory not found: {field_dir}")

        anchors_path = field_dir / "anchors.json"
        vectors_path = field_dir / "vectors.npy"
        basis_path = field_dir / "basis.npy"
        singular_values_path = field_dir / "singular_values.npy"
        metadata_path = field_dir / "metadata.json"
        mean_vector_path = field_dir / "mean_vector.npy"

        for path in [
            anchors_path,
            vectors_path,
            basis_path,
            metadata_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Missing field artifact: {path}")

        anchors_payload = json.loads(
            anchors_path.read_text(encoding="utf-8")
        )

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        vectors = np.load(vectors_path)
        basis = np.load(basis_path)
        if singular_values_path.exists():
            singular_values = np.load(singular_values_path)
        else:
            singular_values = np.array(
                metadata.get("singular_values_top10", []),
                dtype=float,
            )

        mean_vector = None
        if mean_vector_path.exists():
            mean_vector = np.load(mean_vector_path)

        anchors = anchors_payload["anchors"]
        field_name = anchors_payload.get("field", name)

        return ContextMatrix(
            name=field_name,
            anchors=anchors,
            vectors=vectors,
            basis=basis,
            singular_values=singular_values,
            centered=bool(metadata.get("centered", False)),
            mean_vector=mean_vector,
        )

    def load_all(self) -> LoadedAtlas:
        if not self.fields_dir.exists():
            raise FileNotFoundError(f"Fields directory not found: {self.fields_dir}")

        fields = {}
        metadata = {}

        for field_dir in sorted(self.fields_dir.iterdir()):
            if not field_dir.is_dir():
                continue

            metadata_path = field_dir / "metadata.json"

            if not metadata_path.exists():
                continue

            field = self.load_field(field_dir.name)

            fields[field.name] = field
            metadata[field.name] = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )

        if not fields:
            raise ValueError(f"No valid semantic fields found in {self.fields_dir}")

        return LoadedAtlas(fields=fields, metadata=metadata)