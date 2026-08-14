"""McNemar's test on leakage rates: Markdown vs Plain.

Among paired TP traces (same question, both formats correctly refused, both
have leakage classification), tests whether leakage rates differ by format.

Usage:
    uv run python -m analysis.leakage_mcnemar
"""

import pandas as pd
from scipy.stats import binomtest

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


def _mcnemar_test(
    *,
    b: int,
    c: int,
) -> tuple[float, str]:
    """Run McNemar's test. Returns (p_value, method_name).

    Uses exact binomial when discordant pairs < 25, chi-squared otherwise.
    b = MD leaked & PL clean (MD worse).
    c = PL leaked & MD clean (PL worse).
    """
    n_disc = b + c
    if n_disc == 0:
        return 1.0, "no discordant pairs"

    if n_disc < 25:
        p = binomtest(b, n_disc, 0.5).pvalue
        return p, "exact binomial"
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        from scipy.stats import chi2 as chi2_dist

        p = 1 - chi2_dist.cdf(chi2, df=1)
        return p, "chi-squared (Yates)"


def _run_leakage_mcnemar(*, model_size: str) -> None:
    """Run McNemar's test on leakage for one model."""
    df = _load_tp_with_leakage(model_size=model_size)

    md = df[df["prompt_format"] == "markdown"].set_index("question_id")
    pl = df[df["prompt_format"] == "plain"].set_index("question_id")

    paired_ids = md.index.intersection(pl.index)
    print(f"\n{'=' * 60}")
    print(f"  {model_size.upper()} MODEL — McNemar's Test on Leakage")
    print(f"{'=' * 60}")
    print(f"  Paired TP questions (both formats TP with leakage data): {len(paired_ids)}")

    md_leaked = md.loc[paired_ids, "has_leakage"].astype(bool)
    pl_leaked = pl.loc[paired_ids, "has_leakage"].astype(bool)

    # Contingency table for leakage
    both_leak = (md_leaked & pl_leaked).sum()
    md_only = (md_leaked & ~pl_leaked).sum()
    pl_only = (~md_leaked & pl_leaked).sum()
    neither = (~md_leaked & ~pl_leaked).sum()

    print("\n  Leakage contingency table:")
    print(f"  {'':20s} PL leaked    PL clean")
    print(f"  {'MD leaked':20s} {both_leak:8d}    {md_only:8d}")
    print(f"  {'MD clean':20s} {pl_only:8d}    {neither:8d}")

    print(f"\n  Discordant pairs: {md_only + pl_only}")
    print(f"    b (MD leaked, PL clean): {md_only}")
    print(f"    c (PL leaked, MD clean): {pl_only}")

    md_rate = 100 * md_leaked.sum() / len(paired_ids)
    pl_rate = 100 * pl_leaked.sum() / len(paired_ids)
    print("\n  Leakage rates (paired subset):")
    print(f"    MD: {md_leaked.sum()}/{len(paired_ids)} = {md_rate:.1f}%")
    print(f"    PL: {pl_leaked.sum()}/{len(paired_ids)} = {pl_rate:.1f}%")
    print(f"    Delta: {md_rate - pl_rate:+.1f} pp")

    p, method = _mcnemar_test(b=int(md_only), c=int(pl_only))
    sig = "*" if p < 0.05 else ""
    print(f"\n  McNemar's test ({method}):")
    print(f"    p = {p:.4f} {sig}")
    if p < 0.05:
        if md_only > pl_only:
            print("    Direction: Markdown leaks MORE than Plain")
        else:
            print("    Direction: Plain leaks MORE than Markdown")
    else:
        print("    No significant difference in leakage rates between formats")


def main() -> None:
    """Run leakage McNemar's test for all models."""
    print("McNEMAR'S TEST ON LEAKAGE RATES — MD vs Plain")
    print("Tests whether format affects leakage among correct refusals")

    for model_size in MODEL_SIZES:
        _run_leakage_mcnemar(model_size=model_size)


if __name__ == "__main__":
    main()
