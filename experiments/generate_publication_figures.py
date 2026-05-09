from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from atlas.core import SemanticFieldLoader


BASE_DIR = Path(__file__).resolve().parent.parent
FIELDS_DIR = BASE_DIR / "fields"
RESULTS_PATH = BASE_DIR / "datasets" / "ace_runtime_benchmark_results_v1.csv"
POLICY_PATH = BASE_DIR / "datasets" / "runtime_policy_benchmark_results.csv"
VECTORS_PATH = BASE_DIR / "datasets" / "benchmark_vectors.npy"
OUTPUT_DIR = BASE_DIR / "outputs" / "paper_figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str):
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


def load_anchor_data():
    atlas = SemanticFieldLoader(FIELDS_DIR).load_all()

    rows = []
    for field_name, field in atlas.fields.items():
        for anchor, vector in zip(field.anchors, field.vectors):
            rows.append({
                "field": field_name,
                "text": anchor,
                "vector": vector,
            })

    return pd.DataFrame(rows)


def prepare_pca():
    anchors = load_anchor_data()
    benchmark = pd.read_csv(RESULTS_PATH)
    benchmark_vectors = np.load(VECTORS_PATH)

    anchor_vectors = np.vstack(anchors["vector"].to_numpy())
    all_vectors = np.vstack([anchor_vectors, benchmark_vectors])

    pca = PCA(n_components=2)
    points = pca.fit_transform(all_vectors)

    anchors["x"] = points[:len(anchors), 0]
    anchors["y"] = points[:len(anchors), 1]

    benchmark = benchmark.copy()
    benchmark["x"] = points[len(anchors):, 0]
    benchmark["y"] = points[len(anchors):, 1]

    benchmark["stability_index"] = (
        benchmark["field_margin"]
        * benchmark["best_density"]
        / benchmark["best_cost"]
    )

    return anchors, benchmark


def figure_1_fields(anchors):
    plt.figure(figsize=(8, 6))

    for field, group in anchors.groupby("field"):
        plt.scatter(group["x"], group["y"], s=18, alpha=0.55, label=field)

        cx = group["x"].mean()
        cy = group["y"].mean()
        plt.scatter(cx, cy, marker="X", s=120, edgecolors="black")
        plt.text(cx, cy, field, fontsize=9, weight="bold")

    plt.title("Semantic Field Geometry")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(fontsize=8)

    savefig("figure_01_semantic_field_geometry.png")


def figure_2_benchmark_overlay(anchors, benchmark):
    plt.figure(figsize=(8, 6))

    for field, group in anchors.groupby("field"):
        plt.scatter(group["x"], group["y"], s=12, alpha=0.12)

    for label, group in benchmark.groupby("label"):
        plt.scatter(group["x"], group["y"], s=42, marker="x", label=label)

    plt.title("Benchmark Examples Over Semantic Fields")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(fontsize=7, ncol=2)

    savefig("figure_02_benchmark_overlay.png")


def figure_3_stability_overlay(anchors, benchmark):
    plt.figure(figsize=(8, 6))

    for _, group in anchors.groupby("field"):
        plt.scatter(group["x"], group["y"], s=12, alpha=0.10)

    scatter = plt.scatter(
        benchmark["x"],
        benchmark["y"],
        c=benchmark["stability_index"],
        s=48,
        marker="o",
    )

    plt.colorbar(scatter, label="Stability Index")
    plt.title("Semantic Stability Overlay")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    savefig("figure_03_stability_overlay.png")


def figure_4_dispersion_regions(anchors, benchmark):
    unstable_labels = {
        "low_context",
        "nonsense",
        "contradiction",
        "adversarial",
        "incomplete",
    }

    focus = benchmark[benchmark["label"].isin(unstable_labels)]

    plt.figure(figsize=(8, 6))

    for field, group in anchors.groupby("field"):
        plt.scatter(group["x"], group["y"], s=12, alpha=0.10)

    for label, group in focus.groupby("label"):
        plt.scatter(group["x"], group["y"], s=48, marker="x", label=label)

    plt.title("Semantic Dispersion Regions")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(fontsize=8)

    savefig("figure_04_semantic_dispersion_regions.png")


def figure_5_stability_by_label():
    policy = pd.read_csv(POLICY_PATH)

    summary = (
        policy.groupby("label")["stability_index"]
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(8, 5))
    plt.bar(summary.index, summary.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Average Stability Index by Benchmark Category")
    plt.xlabel("Benchmark Category")
    plt.ylabel("Average Stability Index")

    savefig("figure_05_stability_by_label.png")


def figure_6_runtime_actions():
    policy = pd.read_csv(POLICY_PATH)

    counts = policy["action"].value_counts()

    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values)
    plt.title("Semantic Dispersion Gate Runtime Actions")
    plt.xlabel("Runtime Action")
    plt.ylabel("Count")

    savefig("figure_06_runtime_actions.png")


def main():
    anchors, benchmark = prepare_pca()

    figure_1_fields(anchors)
    figure_2_benchmark_overlay(anchors, benchmark)
    figure_3_stability_overlay(anchors, benchmark)
    figure_4_dispersion_regions(anchors, benchmark)
    figure_5_stability_by_label()
    figure_6_runtime_actions()

    print()
    print("=" * 80)
    print("PUBLICATION FIGURES GENERATED")
    print("=" * 80)


if __name__ == "__main__":
    main()