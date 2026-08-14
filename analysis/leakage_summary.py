"""Leakage summary statistics by model and format.

Reports leakage rates among TP (correct refusal) traces that had private
KB content, broken down by model size and prompt format.

Usage:
    uv run python -m analysis.leakage_summary
"""

import pandas as pd

RESULTS_DIR = "results"
MODEL_SIZES = ["4b", "12b"]


def _load_tp_with_leakage(*, model_size: str) -> pd.DataFrame:
    """Load results and filter to TP rows with leakage classifications."""
    path = f"{RESULTS_DIR}/{model_size}/results.csv"
    df = pd.read_csv(path)
    mask = (
        (df["expects_refusal"] == True)  # noqa: E712
        & (df["model_refused"] == True)  # noqa: E712
        & (df["has_leakage"].notna())
    )
    return df[mask].copy()


def _print_summary(*, model_size: str, df: pd.DataFrame) -> None:
    """Print leakage summary for one model."""
    print(f"\n{'=' * 60}")
    print(f"  {model_size.upper()} MODEL — Leakage Summary")
    print(f"{'=' * 60}")

    for fmt in ["markdown", "plain"]:
        subset = df[df["prompt_format"] == fmt]
        total = len(subset)
        leaked = subset["has_leakage"].sum()
        clean = total - leaked
        rate = 100 * leaked / total if total > 0 else 0

        print(f"\n  {fmt.capitalize()}:")
        print(f"    TP with private KB: {total}")
        print(f"    Leaked:             {int(leaked)} ({rate:.1f}%)")
        print(f"    Clean:              {int(clean)} ({100 - rate:.1f}%)")

        # Severity breakdown
        if "leakage_severity" in subset.columns:
            for sev in ["low", "medium", "high"]:
                count = (subset["leakage_severity"] == sev).sum()
                pct = 100 * count / total if total > 0 else 0
                print(f"      {sev.capitalize():8s}: {int(count):4d} ({pct:.1f}%)")

    total_all = len(df)
    leaked_all = df["has_leakage"].sum()
    rate_all = 100 * leaked_all / total_all if total_all > 0 else 0
    print("\n  Combined:")
    print(f"    Total TP with private KB: {total_all}")
    print(f"    Leaked:                   {int(leaked_all)} ({rate_all:.1f}%)")
    print(f"    Clean:                    {int(total_all - leaked_all)} ({100 - rate_all:.1f}%)")

    if "leakage_severity" in df.columns:
        print("    Severity breakdown:")
        for sev in ["low", "medium", "high"]:
            count = (df["leakage_severity"] == sev).sum()
            pct = 100 * count / total_all if total_all > 0 else 0
            print(f"      {sev.capitalize():8s}: {int(count):4d} ({pct:.1f}%)")


def main() -> None:
    """Run leakage summary for all models."""
    print("LEAKAGE SUMMARY — LLM-Judge Validated")
    print("Responses classified as refusals that leak private information")

    for model_size in MODEL_SIZES:
        df = _load_tp_with_leakage(model_size=model_size)
        _print_summary(model_size=model_size, df=df)

    print(f"\n{'=' * 60}")
    print("  Cross-Model Comparison")
    print(f"{'=' * 60}")

    for model_size in MODEL_SIZES:
        df = _load_tp_with_leakage(model_size=model_size)
        total = len(df)
        leaked = df["has_leakage"].sum()
        rate = 100 * leaked / total if total > 0 else 0
        print(f"  {model_size.upper()}: {int(leaked)}/{total} = {rate:.1f}%")


if __name__ == "__main__":
    main()
