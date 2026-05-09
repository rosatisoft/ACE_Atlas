from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

POLICY_RESULTS = BASE_DIR / "datasets" / "runtime_policy_benchmark_results.csv"
LLM_RESULTS = BASE_DIR / "datasets" / "llm_gate_demo_results.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str):
    csv_path = OUTPUT_DIR / f"{name}.csv"
    md_path = OUTPUT_DIR / f"{name}.md"

    df.to_csv(csv_path, index=True)
    md_path.write_text(df.to_markdown(), encoding="utf-8")

    print("Saved:", csv_path)
    print("Saved:", md_path)


def main():
    policy = pd.read_csv(POLICY_RESULTS)
    llm = pd.read_csv(LLM_RESULTS)

    # 1. Runtime action counts
    action_counts = (
        policy["action"]
        .value_counts()
        .rename_axis("action")
        .to_frame("count")
    )
    save_table(action_counts, "table_01_runtime_action_counts")

    # 2. Action by label matrix
    action_by_label = (
        policy.groupby(["label", "action"])
        .size()
        .unstack(fill_value=0)
    )
    save_table(action_by_label, "table_02_action_by_label")

    # 3. Average stability by label
    stability_by_label = (
        policy.groupby("label")
        .agg(
            count=("id", "count"),
            avg_best_cost=("best_cost", "mean"),
            avg_margin=("field_margin", "mean"),
            avg_density=("best_density", "mean"),
            avg_stability=("stability_index", "mean"),
        )
        .sort_values("avg_stability")
    )
    save_table(stability_by_label, "table_03_stability_by_label")

    # 4. Average stability by action
    stability_by_action = (
        policy.groupby("action")
        .agg(
            count=("id", "count"),
            avg_best_cost=("best_cost", "mean"),
            avg_margin=("field_margin", "mean"),
            avg_density=("best_density", "mean"),
            avg_stability=("stability_index", "mean"),
        )
        .sort_values("avg_stability")
    )
    save_table(stability_by_action, "table_04_stability_by_action")

    # 5. LLM calls by action
    llm_calls_by_action = (
        llm.groupby(["action", "llm_called"])
        .size()
        .unstack(fill_value=0)
    )
    save_table(llm_calls_by_action, "table_05_llm_calls_by_action")

    # 6. Token savings summary
    token_summary = pd.DataFrame({
        "value": {
            "samples": len(llm),
            "llm_calls_executed": int(llm["llm_called"].sum()),
            "llm_calls_prevented": int((~llm["llm_called"]).sum()),
            "baseline_tokens": int(llm["baseline_assumed_tokens"].sum()),
            "gated_tokens": int(llm["gated_tokens"].sum()),
            "estimated_savings": int(llm["estimated_savings"].sum()),
            "estimated_savings_percent": round(
                llm["estimated_savings"].sum()
                / llm["baseline_assumed_tokens"].sum()
                * 100,
                2,
            ),
        }
    })
    save_table(token_summary, "table_06_token_savings_summary")

    # 7. Token savings by label
    savings_by_label = (
        llm.groupby("label")
        .agg(
            count=("id", "count"),
            llm_calls=("llm_called", "sum"),
            baseline_tokens=("baseline_assumed_tokens", "sum"),
            gated_tokens=("gated_tokens", "sum"),
            estimated_savings=("estimated_savings", "sum"),
        )
    )
    savings_by_label["estimated_savings_percent"] = (
        savings_by_label["estimated_savings"]
        / savings_by_label["baseline_tokens"]
        * 100
    ).round(2)

    save_table(savings_by_label, "table_07_token_savings_by_label")

    print()
    print("=" * 80)
    print("PAPER TABLES GENERATED")
    print("=" * 80)


if __name__ == "__main__":
    main()