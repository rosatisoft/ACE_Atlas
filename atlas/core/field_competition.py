from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class FieldCompetitionResult:
    best_field: str
    best_cost: float
    second_field: str
    second_cost: float
    field_margin: float
    costs: Dict[str, float]
    ranking: List[Tuple[str, float]]


def analyze_field_competition(costs: Dict[str, float]) -> FieldCompetitionResult:
    """
    Analyze semantic competition across fields.

    Lower cost means stronger alignment with a field.
    Margin measures how clearly the best field wins over the second-best field.
    """

    if not costs:
        raise ValueError("costs cannot be empty")

    if len(costs) < 2:
        raise ValueError("at least two fields are required for competition")

    ranking = sorted(costs.items(), key=lambda item: item[1])

    best_field, best_cost = ranking[0]
    second_field, second_cost = ranking[1]

    return FieldCompetitionResult(
        best_field=best_field,
        best_cost=float(best_cost),
        second_field=second_field,
        second_cost=float(second_cost),
        field_margin=float(second_cost - best_cost),
        costs=dict(costs),
        ranking=[(name, float(cost)) for name, cost in ranking],
    )