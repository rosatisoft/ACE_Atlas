from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR / "datasets" / "ace_runtime_benchmark_results_v1.csv"
)

OUTPUT_PATH = (
    BASE_DIR / "datasets" / "benchmark_vectors.npy"
)


def embed(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    return np.array(
        response.data[0].embedding,
        dtype=float,
    )


def main():
    df = pd.read_csv(DATASET_PATH)

    vectors = []

    total = len(df)

    print("=" * 80)
    print("CACHING BENCHMARK EMBEDDINGS")
    print("=" * 80)

    for idx, text in enumerate(df["text"], start=1):
        print(f"[{idx}/{total}] {text[:80]}")

        vector = embed(text)

        vectors.append(vector)

    vectors = np.array(vectors)

    np.save(OUTPUT_PATH, vectors)

    print()
    print("Saved:", OUTPUT_PATH)
    print("Shape:", vectors.shape)


if __name__ == "__main__":
    main()