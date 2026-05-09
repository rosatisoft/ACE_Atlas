from dataclasses import dataclass
from typing import Dict, Optional

from .field_competition import FieldCompetitionResult


@dataclass
class StabilityResult:
    stability_index: float
    best_density: float
    density_margin: float
    interpretation: str


def compute_stability_index(
    best_cost: float,
    field_margin: float,
    best_density: float,
) -> float:
    """
    Experimental ACE stability index.

    Current formula:

        stability = (margin * density) / best_cost

    Interpretation:
    - higher margin = clearer field winner
    - higher density = stronger local anchor support
    - lower cost = stronger subspace alignment

    This formula is experimental and may evolve.
    """

    if best_cost <= 0:
        return 0.0

    return float((field_margin * best_density) / best_cost)


def analyze_stability(
    competition: FieldCompetitionResult,
    densities: Dict[str, float],
) -> StabilityResult:
    best_density = float(densities.get(competition.best_field, 0.0))

    second_density = float(densities.get(competition.second_field, 0.0))

    density_margin = best_density - second_density

    stability_index = compute_stability_index(
        best_cost=competition.best_cost,
        field_margin=competition.field_margin,
        best_density=best_density,
    )

    interpretation = interpret_stability(
        stability_index=stability_index,
        best_cost=competition.best_cost,
        field_margin=competition.field_margin,
        best_density=best_density,
    )

    return StabilityResult(
        stability_index=stability_index,
        best_density=best_density,
        density_margin=density_margin,
        interpretation=interpretation,
    )


def interpret_stability(
    stability_index: float,
    best_cost: float,
    field_margin: float,
    best_density: float,
) -> str:
    """
    Provisional interpretation layer.

    This is not a final policy.
    It is meant for observation and research reporting.
    """

    if stability_index < 0.01 and best_cost > 0.75 and field_margin < 0.05:
        return "low_context_or_unstable"

    if stability_index < 0.03 and field_margin < 0.08:
        return "weak_or_competing_context"

    if stability_index < 0.08:
        return "moderate_instability"

    if stability_index < 0.18:
        return "moderate_stability"

    return "high_stability"