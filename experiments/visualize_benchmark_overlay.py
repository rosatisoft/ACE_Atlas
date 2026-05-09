from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from atlas.core import SemanticFieldLoader


BASE_DIR = Path(__file__).resolve().parent.parent
FIELDS_DIR = BASE_DIR / "fields"
RESULTS_PATH = BASE_DIR / "datasets" / "ace_runtime_benchmark_results_v1.csv"
BENCHMARK_VECTORS_PATH = BASE_DIR / "datasets" / "benchmark_vectors.npy"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def load_anchor_rows():
    atlas = SemanticFieldLoader(FIELDS_DIR).load_all()

    rows = []

    for field_name, field in atlas.fields.items():
        for anchor, vector in zip(field.anchors, field.vectors):
            rows.append({
                "kind": "anchor",
                "field": field_name,
                "label": field_name,
                "text": anchor,
                "vector": vector,
            })

    return rows


def main():
    anchor_rows = load_anchor_rows()

    benchmark_df = pd.read_csv(RESULTS_PATH)
    benchmark_vectors = np.load(BENCHMARK_VECTORS_PATH)

    anchor_vectors = np.array([row["vector"] for row in anchor_rows])

    all_vectors = np.vstack([
        anchor_vectors,
        benchmark_vectors,
    ])

    pca = PCA(n_components=2)
    all_points = pca.fit_transform(all_vectors)

    anchor_points = all_points[: len(anchor_vectors)]
    benchmark_points = all_points[len(anchor_vectors):]

    anchor_df = pd.DataFrame({
        "x": anchor_points[:, 0],
        "y": anchor_points[:, 1],
        "field": [row["field"] for row in anchor_rows],
    })

    benchmark_df = benchmark_df.copy()
    benchmark_df["x"] = benchmark_points[:, 0]
    benchmark_df["y"] = benchmark_points[:, 1]

    benchmark_df["stability_index"] = (
        benchmark_df["field_margin"]
        * benchmark_df["best_density"]
        / benchmark_df["best_cost"]
    )

    # ------------------------------------------------------------
    # Plot 1: benchmark labels over field anchors
    # ------------------------------------------------------------
    plt.figure(figsize=(14, 10))

    for field_name, group in anchor_df.groupby("field"):
        plt.scatter(
            group["x"],
            group["y"],
            alpha=0.18,
            s=25,
            label=f"{field_name} anchors",
        )

    for label, group in benchmark_df.groupby("label"):
        plt.scatter(
            group["x"],
            group["y"],
            s=80,
            marker="x",
            label=f"{label} benchmark",
        )

    plt.title("ACE Atlas PCA — Benchmark Examples Over Semantic Fields")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()

    output = OUTPUT_DIR / "atlas_benchmark_overlay_by_label.png"
    plt.savefig(output, dpi=220)
    plt.close()

    print("Saved:", output)

    # ------------------------------------------------------------
    # Plot 2: stability heat overlay
    # ------------------------------------------------------------
    plt.figure(figsize=(14, 10))

    for field_name, group in anchor_df.groupby("field"):
        plt.scatter(
            group["x"],
            group["y"],
            alpha=0.12,
            s=20,
            label=f"{field_name} anchors",
        )

    scatter = plt.scatter(
        benchmark_df["x"],
        benchmark_df["y"],
        c=benchmark_df["stability_index"],
        s=90,
        marker="o",
    )

    plt.colorbar(scatter, label="Stability Index")
    plt.title("ACE Atlas PCA — Benchmark Stability Overlay")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()

    output = OUTPUT_DIR / "atlas_benchmark_overlay_stability.png"
    plt.savefig(output, dpi=220)
    plt.close()

    print("Saved:", output)

    # ------------------------------------------------------------
    # Plot 3: low-context and unstable cases
    # ------------------------------------------------------------
    focus_labels = {
        "low_context",
        "nonsense",
        "contradiction",
        "adversarial",
        "incomplete",
    }

    focus_df = benchmark_df[
        benchmark_df["label"].isin(focus_labels)
    ]

    plt.figure(figsize=(14, 10))

    for field_name, group in anchor_df.groupby("field"):
        plt.scatter(
            group["x"],
            group["y"],
            alpha=0.12,
            s=20,
            label=f"{field_name} anchors",
        )

    for label, group in focus_df.groupby("label"):
        plt.scatter(
            group["x"],
            group["y"],
            s=90,
            marker="x",
            label=label,
        )

    plt.title("ACE Atlas PCA — Low-Context and Unstable Benchmark Regions")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()

    output = OUTPUT_DIR / "atlas_benchmark_overlay_unstable.png"
    plt.savefig(output, dpi=220)
    plt.close()

    print("Saved:", output)


if __name__ == "__main__":
    main()