from dataclasses import dataclass
from typing import Literal

from .runtime_evaluator import RuntimeEvaluation


RuntimeAction = Literal[
    "ALLOW",
    "ALLOW_LIGHT",
    "CLARIFY",
    "ABSTAIN",
]


@dataclass
class RuntimePolicy:
    name: str = "default"

    high_stability_threshold: float = 0.18
    medium_stability_threshold: float = 0.06
    low_stability_threshold: float = 0.01

    high_margin_threshold: float = 0.20
    medium_margin_threshold: float = 0.08

    high_density_threshold: float = 0.30
    low_density_threshold: float = 0.18

    high_cost_threshold: float = 0.75


@dataclass
class RuntimePolicyDecision:
    action: RuntimeAction
    route: str
    confidence: float
    reason: str


DEFAULT_POLICY = RuntimePolicy()


def decide_runtime_action(
    evaluation: RuntimeEvaluation,
    policy: RuntimePolicy = DEFAULT_POLICY,
) -> RuntimePolicyDecision:
    """
    Decide what the runtime should do before invoking an LLM.

    This is the Semantic Dispersion Gate policy.

    It does not decide truth.
    It decides whether the input is semantically stable enough
    to proceed into reasoning or generation.
    """

    stability = evaluation.stability_index
    margin = evaluation.field_margin
    density = evaluation.stability.best_density
    cost = evaluation.best_cost

    if (
        stability < policy.low_stability_threshold
        and cost > policy.high_cost_threshold
        and margin < policy.medium_margin_threshold
        and density < policy.low_density_threshold
    ):
        return RuntimePolicyDecision(
            action="CLARIFY",
            route="none",
            confidence=0.90,
            reason="insufficient_context_high_dispersion",
        )

    if (
        stability < policy.low_stability_threshold
        and margin < policy.medium_margin_threshold
    ):
        return RuntimePolicyDecision(
            action="CLARIFY",
            route="none",
            confidence=0.80,
            reason="weak_contextual_determination",
        )

    if (
        stability >= policy.high_stability_threshold
        and margin >= policy.high_margin_threshold
        and density >= policy.low_density_threshold
    ):
        return RuntimePolicyDecision(
            action="ALLOW",
            route=evaluation.best_field,
            confidence=0.90,
            reason="stable_contextual_field",
        )

    if (
        stability >= policy.medium_stability_threshold
        and margin >= policy.medium_margin_threshold
    ):
        return RuntimePolicyDecision(
            action="ALLOW_LIGHT",
            route=evaluation.best_field,
            confidence=0.72,
            reason="moderately_stable_context",
        )

    if (
        cost > policy.high_cost_threshold
        and margin < policy.medium_margin_threshold
    ):
        return RuntimePolicyDecision(
            action="CLARIFY",
            route="none",
            confidence=0.75,
            reason="high_cost_low_margin",
        )

    return RuntimePolicyDecision(
        action="CLARIFY",
        route="none",
        confidence=0.60,
        reason="semantic_dispersion_detected",
    )