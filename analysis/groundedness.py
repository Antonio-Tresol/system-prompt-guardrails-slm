"""Groundedness analysis: paired comparison of hallucination rates.

Reports groundedness rates for compliance responses and runs McNemar's
test comparing Markdown vs Plain on questions where both formats complied.

Usage:
    uv run python -m analysis.groundedness
"""

import pandas as pd
from scipy.stats import binomtest, chi2

RESULTS_DIR = "results"
MODEL_SIZES = ["4b", "12b"]
ALPHA = 0.05


def _mcnemar_test(
    *,
    md_correct: pd.Series,
    plain_correct: pd.Series,
) -> dict:
    """McNemar's test for paired binary outcomes."""
    b = int((md_correct & ~plain_correct).sum())
    c = int((~md_correct & plain_correct).sum())
    n_discordant = b + c

    if n_discordant == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "method": "none"}

    if n_discordant < 25:
        p_value = binomtest(b, n_discordant, 0.5).pvalue
        method = "exact binomial"
    else:
        stat = (b - c) ** 2 / n_discordant
        p_value = 1 - chi2.cdf(stat, df=1)
        method = "chi-squared"

    return {"b": b, "c": c, "n_discordant": n_discordant, "p_value": p_value, "method": method}


def _run_model(*, model_size: str) -> None:
    """Run groundedness analysis for one model."""
    path = f"{RESULTS_DIR}/{model_size}/results.csv"
    df = pd.read_csv(path)

    print(f"\n{'=' * 60}")
    print(f"  {model_size.upper()} MODEL — Groundedness Analysis")
    print(f"{'=' * 60}")

    complied = df[df["model_refused"] == False]  # noqa: E712

    for fmt in ["markdown", "plain"]:
        subset = complied[complied["prompt_format"] == fmt]
        total = len(subset)
        grounded = int(subset["is_grounded"].eq(True).sum())
        hallucinated = int(subset["is_grounded"].eq(False).sum())
        missing = int(subset["is_grounded"].isna().sum())

        print(f"\n  {fmt.capitalize()} compliance responses: {total}")
        print(f"    Grounded:     {grounded} ({100 * grounded / total:.1f}%)" if total else "")
        hall_pct = 100 * hallucinated / total if total else 0
        print(f"    Hallucinated: {hallucinated} ({hall_pct:.1f}%)" if total else "")
        if missing > 0:
            print(f"    Missing:      {missing}")

    # Paired groundedness: questions where BOTH formats complied
    md = df[df["prompt_format"] == "markdown"].set_index("question_id").sort_index()
    pl = df[df["prompt_format"] == "plain"].set_index("question_id").sort_index()

    pd.set_option("future.no_silent_downcasting", True)
    md_refused = md["model_refused"].fillna(value=False).astype(bool)
    pl_refused = pl["model_refused"].fillna(value=False).astype(bool)
    both_complied = (~md_refused) & (~pl_refused)
    n_both = int(both_complied.sum())

    print(f"\n  Paired analysis (both formats complied): {n_both} questions")

    if n_both > 0:
        md_grounded = md.loc[both_complied, "is_grounded"].eq(True)
        pl_grounded = pl.loc[both_complied, "is_grounded"].eq(True)

        test = _mcnemar_test(md_correct=md_grounded, plain_correct=pl_grounded)
        sig = " *" if test["p_value"] < ALPHA else ""

        print(f"    b (MD grounded, PL not): {test['b']}")
        print(f"    c (PL grounded, MD not): {test['c']}")
        print(f"    McNemar's ({test['method']}): p={test['p_value']:.4f}{sig}")

    # Retry count distribution
    print("\n  Retry count distribution:")
    retry_counts = df["retry_count"].value_counts().sort_index()
    for count, n in retry_counts.items():
        print(f"    {count} retries: {n} runs")


def main() -> None:
    """Run groundedness analysis for all models."""
    print("GROUNDEDNESS ANALYSIS — Hallucination Detection")
    print("Checks whether compliance responses are grounded in KB sources")

    for model_size in MODEL_SIZES:
        _run_model(model_size=model_size)


if __name__ == "__main__":
    main()
