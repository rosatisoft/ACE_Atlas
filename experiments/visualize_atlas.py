from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from atlas.core import SemanticFieldLoader


BASE_DIR = Path(__file__).resolve().parent.parent
FIELDS_DIR = BASE_DIR / "fields"
DATASET_RESULTS = BASE_DIR / "datasets" / "ace_runtime_benchmark_results_v1.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def load_anchor_vectors():
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
                "stability": None,
            })

    return rows


def plot_fields(anchor_rows):
    vectors = np.array([row["vector"] for row in anchor_rows])
    pca = PCA(n_components=2)
    points = pca.fit_transform(vectors)

    df = pd.DataFrame({
        "x": points[:, 0],
        "y": points[:, 1],
        "field": [row["field"] for row in anchor_rows],
        "text": [row["text"] for row in anchor_rows],
    })

    plt.figure(figsize=(12, 9))

    for field_name, group in df.groupby("field"):
        plt.scatter(
            group["x"],
            group["y"],
            label=field_name,
            alpha=0.65,
            s=35,
        )

        centroid_x = group["x"].mean()
        centroid_y = group["y"].mean()

        plt.scatter(
            centroid_x,
            centroid_y,
            marker="X",
            s=220,
            edgecolors="black",
        )

        plt.text(
            centroid_x,
            centroid_y,
            field_name,
            fontsize=11,
            weight="bold",
        )

    plt.title("ACE Atlas PCA Map — Semantic Field Anchors")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()

    output = OUTPUT_DIR / "atlas_pca_fields.png"
    plt.savefig(output, dpi=200)
    plt.close()

    print("Saved:", output)


def plot_benchmark_overlay(anchor_rows):
    if not DATASET_RESULTS.exists():
        print("Benchmark CSV not found:", DATASET_RESULTS)
        return

    atlas = SemanticFieldLoader(FIELDS_DIR).load_all()
    benchmark = pd.read_csv(DATASET_RESULTS)

    anchor_vectors = np.array([row["vector"] for row in anchor_rows])

    pca = PCA(n_components=2)
    anchor_points = pca.fit_transform(anchor_vectors)

    anchor_df = pd.DataFrame({
        "x": anchor_points[:, 0],
        "y": anchor_points[:, 1],
        "field": [row["field"] for row in anchor_rows],
    })

    plt.figure(figsize=(13, 10))

    for field_name, group in anchor_df.groupby("field"):
        plt.scatter(
            group["x"],
            group["y"],
            label=f"{field_name} anchors",
            alpha=0.25,
            s=25,
        )

    # NOTE:
    # The benchmark CSV currently does not store vectors.
    # So this plot uses stability categories only as a reporting scaffold.
    # Full benchmark embedding overlay will be added once benchmark vectors are cached.

    label_counts = benchmark["label"].value_counts()

    text = "\n".join(
        f"{label}: {count}"
        for label, count in label_counts.items()
    )

    plt.text(
        0.02,
        0.98,
        text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    plt.title("ACE Atlas PCA Map — Field Anchors with Benchmark Summary")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()

    output = OUTPUT_DIR / "atlas_pca_benchmark_summary.png"
    plt.savefig(output, dpi=200)
    plt.close()

    print("Saved:", output)


def plot_stability_summary():
    if not DATASET_RESULTS.exists():
        print("Benchmark CSV not found:", DATASET_RESULTS)
        return

    df = pd.read_csv(DATASET_RESULTS)

    if "best_density" not in df.columns:
        print("best_density column not found. Run benchmark first.")
        return

    df["stability_index"] = (
        df["field_margin"] * df["best_density"] / df["best_cost"]
    )

    summary = (
        df.groupby("label")["stability_index"]
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(12, 7))
    plt.bar(summary.index, summary.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("ACE Atlas — Average Stability Index by Label")
    plt.xlabel("Benchmark Label")
    plt.ylabel("Average Stability Index")
    plt.tight_layout()

    output = OUTPUT_DIR / "atlas_stability_by_label.png"
    plt.savefig(output, dpi=200)
    plt.close()

    print("Saved:", output)


def main():
    anchor_rows = load_anchor_vectors()

    print("=" * 80)
    print("ACE ATLAS VISUALIZATION")
    print("=" * 80)
    print("Anchor vectors:", len(anchor_rows))

    plot_fields(anchor_rows)
    plot_benchmark_overlay(anchor_rows)
    plot_stability_summary()


if __name__ == "__main__":
    main()