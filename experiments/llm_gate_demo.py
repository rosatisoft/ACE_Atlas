from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

from atlas.core import (
    AtlasRuntimeEvaluator,
    SemanticFieldLoader,
    decide_runtime_action,
)


BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    BASE_DIR / "datasets" / "ace_runtime_benchmark_results_v1.csv"
)

VECTORS_PATH = (
    BASE_DIR / "datasets" / "benchmark_vectors.npy"
)

OUTPUT_PATH = (
    BASE_DIR / "datasets" / "llm_gate_demo_results.csv"
)

client = OpenAI()


def estimate_tokens(text: str) -> int:
    """
    Simple token approximation for early experiments.
    Conservative estimate: 1 token ≈ 4 characters.
    """

    return max(1, len(text) // 4)


def clarify_response(reason: str) -> str:
    responses = {
        "insufficient_context_high_dispersion": (
            "I need a little more context before I can answer accurately. "
            "Could you clarify what you want me to focus on?"
        ),
        "weak_contextual_determination": (
            "Your request can be interpreted in more than one way. "
            "Could you provide more specific context?"
        ),
        "high_cost_low_margin": (
            "I do not have enough stable context to answer reliably. "
            "Please add more details or clarify the intended domain."
        ),
        "semantic_dispersion_detected": (
            "I need more information before giving a reliable answer."
        ),
    }

    return responses.get(
        reason,
        "I need more context before giving a reliable answer.",
    )


def call_llm(text: str, mode: str) -> tuple[str, int]:
    if mode == "ALLOW_LIGHT":
        max_tokens = 80
        system = (
            "Answer briefly and only within the most likely context. "
            "Do not over-explain."
        )
    else:
        max_tokens = 250
        system = (
            "Answer clearly and directly within the detected context."
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )

    content = response.choices[0].message.content or ""

    # Prefer actual usage if available.
    if response.usage:
        total_tokens = response.usage.total_tokens
    else:
        total_tokens = estimate_tokens(system + text + content)

    return content.strip(), total_tokens


def main():
    atlas = SemanticFieldLoader(BASE_DIR / "fields").load_all()
    evaluator = AtlasRuntimeEvaluator(atlas)

    df = pd.read_csv(RESULTS_PATH)
    vectors = np.load(VECTORS_PATH)

    rows = []

    print("=" * 80)
    print("ACE ATLAS — LLM GATE DEMO")
    print("=" * 80)

    for idx, row in df.iterrows():
        text = row["text"]
        vector = vectors[idx]

        evaluation = evaluator.evaluate(
            text=text,
            vector=vector,
        )

        decision = decide_runtime_action(evaluation)

        baseline_prompt_tokens = estimate_tokens(text)
        baseline_assumed_tokens = baseline_prompt_tokens + 250

        if decision.action == "CLARIFY":
            answer = clarify_response(decision.reason)
            gated_tokens = estimate_tokens(text + answer)
            llm_called = False
        else:
            answer, gated_tokens = call_llm(text, decision.action)
            llm_called = True

        estimated_savings = baseline_assumed_tokens - gated_tokens

        rows.append({
            "id": row["id"],
            "label": row["label"],
            "text": text,
            "best_field": evaluation.best_field,
            "best_cost": evaluation.best_cost,
            "field_margin": evaluation.field_margin,
            "best_density": evaluation.stability.best_density,
            "stability_index": evaluation.stability_index,
            "action": decision.action,
            "route": decision.route,
            "reason": decision.reason,
            "llm_called": llm_called,
            "baseline_assumed_tokens": baseline_assumed_tokens,
            "gated_tokens": gated_tokens,
            "estimated_savings": estimated_savings,
            "answer": answer,
        })

        print(
            f"{idx + 1:03d} | {row['label']:13s} | "
            f"{decision.action:11s} | "
            f"field={evaluation.best_field:12s} | "
            f"stability={evaluation.stability_index:.4f} | "
            f"savings={estimated_savings}"
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("Saved:", OUTPUT_PATH)
    print("Rows:", len(out))
    print()

    print("Actions:")
    print(out["action"].value_counts().to_dict())
    print()

    print("LLM calls:")
    print(out["llm_called"].value_counts().to_dict())
    print()

    print("Estimated baseline tokens:", int(out["baseline_assumed_tokens"].sum()))
    print("Estimated gated tokens:", int(out["gated_tokens"].sum()))
    print("Estimated savings:", int(out["estimated_savings"].sum()))

    savings_pct = (
        out["estimated_savings"].sum()
        / out["baseline_assumed_tokens"].sum()
        * 100
    )

    print("Estimated savings %:", round(savings_pct, 2))


if __name__ == "__main__":
    main()