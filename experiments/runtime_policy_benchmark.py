from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

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
    BASE_DIR / "datasets" / "runtime_policy_benchmark_results.csv"
)


def main():
    atlas = SemanticFieldLoader(BASE_DIR / "fields").load_all()
    evaluator = AtlasRuntimeEvaluator(atlas)

    df = pd.read_csv(RESULTS_PATH)
    vectors = np.load(VECTORS_PATH)

    rows = []

    for idx, row in df.iterrows():
        text = row["text"]
        vector = vectors[idx]

        evaluation = evaluator.evaluate(
            text=text,
            vector=vector,
        )

        decision = decide_runtime_action(evaluation)

        rows.append({
            "id": row["id"],
            "text": text,
            "label": row["label"],
            "best_field": evaluation.best_field,
            "second_field": evaluation.second_field,
            "best_cost": evaluation.best_cost,
            "field_margin": evaluation.field_margin,
            "best_density": evaluation.stability.best_density,
            "stability_index": evaluation.stability_index,
            "interpretation": evaluation.interpretation,
            "action": decision.action,
            "route": decision.route,
            "confidence": decision.confidence,
            "reason": decision.reason,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)

    print("=" * 80)
    print("ACE ATLAS RUNTIME POLICY BENCHMARK")
    print("=" * 80)
    print("Rows:", len(out))
    print("Saved:", OUTPUT_PATH)
    print()

    print("Actions:")
    print(dict(Counter(out["action"])))
    print()

    print("Actions by label:")
    grouped = (
        out.groupby(["label", "action"])
        .size()
        .unstack(fill_value=0)
    )
    print(grouped)
    print()

    print("Average stability by action:")
    print(
        out.groupby("action")["stability_index"]
        .mean()
        .sort_values()
    )


if __name__ == "__main__":
    main()