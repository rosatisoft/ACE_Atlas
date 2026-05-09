import numpy as np
from openai import OpenAI

from atlas.core import (
    AtlasRuntimeEvaluator,
    SemanticFieldLoader,
)

client = OpenAI()


def embed(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    return np.array(response.data[0].embedding, dtype=float)


def main():
    atlas = SemanticFieldLoader("fields").load_all()

    evaluator = AtlasRuntimeEvaluator(atlas)

    print("=" * 80)
    print("ACE ATLAS RUNTIME DEMO")
    print("=" * 80)

    text = "The attorney filed a motion before the hearing"

    print(f"TEXT: {text}")
    print()

    vector = embed(text)

    result = evaluator.evaluate(
        text=text,
        vector=vector,
    )

    print("FIELD COSTS")
    print("-" * 80)

    for name, cost in sorted(result.costs.items(), key=lambda item: item[1]):
        density = result.densities[name]

        print(
            f"{name:15s} "
            f"cost={cost:.6f} "
            f"density={density:.6f}"
        )

    print()
    print("BEST FIELD")
    print("-" * 80)

    print(f"best_field     : {result.best_field}")
    print(f"best_cost      : {result.best_cost:.6f}")
    print(f"second_field   : {result.second_field}")
    print(f"second_cost    : {result.second_cost:.6f}")
    print(f"field_margin   : {result.field_margin:.6f}")

    print()
    print("STABILITY")
    print("-" * 80)

    print(f"stability      : {result.stability_index:.6f}")
    print(f"interpretation : {result.interpretation}")


if __name__ == "__main__":
    main()