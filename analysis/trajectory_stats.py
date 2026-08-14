"""Trajectory and efficiency statistics: Wilcoxon signed-rank tests.

Compares Markdown vs Plain on agent trajectory length, tool calls,
token usage, and duration using paired non-parametric tests.

Usage:
    uv run python -m analysis.trajectory_stats
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = "results"
MODEL_SIZES = ["4b", "12b"]
ALPHA = 0.05


def _run_model(*, model_size: str) -> None:
    """Run trajectory and efficiency analysis for one model."""
    path = f"{RESULTS_DIR}/{model_size}/results.csv"
    df = pd.read_csv(path)

    md = df[df["prompt_format"] == "markdown"].set_index("question_id").sort_index()
    pl = df[df["prompt_format"] == "plain"].set_index("question_id").sort_index()

    print(f"\n{'=' * 60}")
    print(f"  {model_size.upper()} MODEL — Trajectory & Efficiency")
    print(f"{'=' * 60}")

    # Descriptive stats
    header = (
        f"\n  {'Metric':<25s} {'MD median':>10s} {'PL median':>10s}"
        f" {'MD mean':>10s} {'PL mean':>10s}"
    )
    print(header)
    print("  " + "-" * 65)

    for col, label in [
        ("num_steps", "Agent steps"),
        ("num_tool_calls", "Tool calls"),
        ("total_input_tokens", "Input tokens"),
        ("total_output_tokens", "Output tokens"),
        ("duration_ms", "Duration (ms)"),
    ]:
        if col not in md.columns:
            continue
        md_med = np.median(md[col].dropna())
        pl_med = np.median(pl[col].dropna())
        md_mean = md[col].mean()
        pl_mean = pl[col].mean()
        print(f"  {label:<25s} {md_med:>10.0f} {pl_med:>10.0f} {md_mean:>10.1f} {pl_mean:>10.1f}")

    # Wilcoxon signed-rank tests
    print("\n  Wilcoxon Signed-Rank Tests (paired by question):")
    print(f"  {'Metric':<25s} {'W':>10s} {'p':>10s} {'sig':>5s}")
    print("  " + "-" * 55)

    for col, label in [
        ("num_steps", "Agent steps"),
        ("num_tool_calls", "Tool calls"),
        ("total_input_tokens", "Input tokens"),
        ("total_output_tokens", "Output tokens"),
        ("duration_ms", "Duration (ms)"),
    ]:
        if col not in md.columns:
            continue
        md_vals = md[col].values
        pl_vals = pl[col].values
        diff = md_vals - pl_vals
        n_nonzero = (diff != 0).sum()

        if n_nonzero > 0:
            stat, p = wilcoxon(md_vals, pl_vals)
            sig = "*" if p < ALPHA else ""
            print(f"  {label:<25s} {stat:>10.0f} {p:>10.4f} {sig:>5s}")
        else:
            print(f"  {label:<25s} {'—':>10s} {'—':>10s} {'':>5s}  (all pairs identical)")

    # Tool usage patterns
    if "tool_names" in df.columns:
        print("\n  Tool usage patterns:")
        for fmt in ["markdown", "plain"]:
            subset = df[df["prompt_format"] == fmt]
            tool_series = subset["tool_names"].dropna()
            total = len(tool_series)
            used_think = tool_series.str.contains("think").sum()
            used_search = tool_series.str.contains("search").sum()
            think_pct = 100 * used_think / total
            search_pct = 100 * used_search / total
            print(
                f"    {fmt.capitalize()}: think={used_think}/{total}"
                f" ({think_pct:.0f}%), "
                f"search={used_search}/{total} ({search_pct:.0f}%)"
            )


def main() -> None:
    """Run trajectory analysis for all models."""
    print("TRAJECTORY & EFFICIENCY — Wilcoxon Signed-Rank Tests")
    print("Paired non-parametric comparison of agent behavior metrics")

    for model_size in MODEL_SIZES:
        _run_model(model_size=model_size)


if __name__ == "__main__":
    main()
