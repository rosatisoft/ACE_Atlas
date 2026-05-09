from dataclasses import dataclass
from typing import Dict

import numpy as np

from .density import atlas_density
from .field_competition import (
    FieldCompetitionResult,
    analyze_field_competition,
)
from .field_loader import LoadedAtlas
from .stability import StabilityResult, analyze_stability


@dataclass
class RuntimeEvaluation:
    text: str

    costs: Dict[str, float]
    densities: Dict[str, float]

    competition: FieldCompetitionResult
    stability: StabilityResult

    best_field: str
    best_cost: float
    second_field: str
    second_cost: float

    field_margin: float
    stability_index: float

    interpretation: str


class AtlasRuntimeEvaluator:
    """
    Main ACE Atlas runtime evaluator.

    Responsibilities:
    - field competition
    - density analysis
    - stability evaluation
    - unified runtime interpretation

    This module contains no routing policy.
    It only evaluates semantic geometry.
    """

    def __init__(self, atlas: LoadedAtlas):
        self.atlas = atlas

    def evaluate(
        self,
        text: str,
        vector: np.ndarray,
    ) -> RuntimeEvaluation:
        costs = {
            field_name: field.origin_cost(vector)
            for field_name, field in self.atlas.fields.items()
        }

        densities = atlas_density(
            self.atlas.fields,
            vector,
        )

        competition = analyze_field_competition(costs)

        stability = analyze_stability(
            competition=competition,
            densities=densities,
        )

        return RuntimeEvaluation(
            text=text,

            costs=costs,
            densities=densities,

            competition=competition,
            stability=stability,

            best_field=competition.best_field,
            best_cost=competition.best_cost,

            second_field=competition.second_field,
            second_cost=competition.second_cost,

            field_margin=competition.field_margin,

            stability_index=stability.stability_index,

            interpretation=stability.interpretation,
        )