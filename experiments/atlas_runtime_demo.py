import numpy as np
from openai import OpenAI

from atlas.core import (
    SemanticFieldLoader,
    analyze_field_competition,
    analyze_stability,
    atlas_density,
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

    print("=" * 80)
    print("ACE ATLAS RUNTIME DEMO")
    print("=" * 80)

    text = "The attorney filed a motion before the hearing"

    print(f"TEXT: {text}")
    print()

    vector = embed(text)

    costs = {
        field_name: field.origin_cost(vector)
        for field_name, field in atlas.fields.items()
    }

    densities = atlas_density(
        atlas.fields,
        vector,
    )

    competition = analyze_field_competition(costs)

    stability = analyze_stability(
        competition=competition,
        densities=densities,
    )

    print("FIELD COSTS")
    print("-" * 80)

    for name, cost in sorted(costs.items(), key=lambda item: item[1]):
        density = densities[name]

        print(
            f"{name:15s} "
            f"cost={cost:.6f} "
            f"density={density:.6f}"
        )

    print()
    print("BEST FIELD")
    print("-" * 80)

    print(f"best_field     : {competition.best_field}")
    print(f"best_cost      : {competition.best_cost:.6f}")
    print(f"second_field   : {competition.second_field}")
    print(f"second_cost    : {competition.second_cost:.6f}")
    print(f"field_margin   : {competition.field_margin:.6f}")

    print()
    print("STABILITY")
    print("-" * 80)

    print(f"stability      : {stability.stability_index:.6f}")
    print(f"best_density   : {stability.best_density:.6f}")
    print(f"density_margin : {stability.density_margin:.6f}")
    print(f"interpretation : {stability.interpretation}")


if __name__ == "__main__":
    main()