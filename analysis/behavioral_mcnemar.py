"""Primary behavioral analysis: McNemar's test on refusal and compliance rates.

Runs McNemar's paired test comparing Markdown vs Plain prompt formats
on refusal rate (private questions) and compliance rate (public questions),
per model size.

Usage:
    uv run python -m analysis.behavioral_mcnemar
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
    """McNemar's test for paired binary outcomes.

    Uses exact binomial when discordant count < 25, chi-squared otherwise.

    Args:
        md_correct: Boolean series — True if markdown was correct.
        plain_correct: Boolean series — True if plain was correct.

    Returns:
        Dict with counts, test statistic, p-value, and significance.
    """
    b = int((md_correct & ~plain_correct).sum())
    c = int((~md_correct & plain_correct).sum())
    n_discordant = b + c

    if n_discordant == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "method": "no discordant pairs"}

    if n_discordant < 25:
        p_value = binomtest(b, n_discordant, 0.5).pvalue
        method = "exact binomial"
    else:
        stat = (abs(b - c) - 1) ** 2 / n_discordant
        p_value = 1 - chi2.cdf(stat, df=1)
        method = "chi-squared (Yates corrected)"

    return {
        "b": b,
        "c": c,
        "n_discordant": n_discordant,
        "p_value": p_value,
        "method": method,
    }


def _run_model(*, model_size: str) -> None:
    """Run refusal and compliance McNemar's for one model."""
    path = f"{RESULTS_DIR}/{model_size}/results.csv"
    df = pd.read_csv(path)
    df["correct"] = df["expects_refusal"] == df["model_refused"]

    md = df[df["prompt_format"] == "markdown"].set_index("question_id").sort_index()
    pl = df[df["prompt_format"] == "plain"].set_index("question_id").sort_index()

    print(f"\n{'=' * 60}")
    print(f"  {model_size.upper()} MODEL — Behavioral McNemar's Tests")
    print(f"{'=' * 60}")

    for label, is_private in [("REFUSAL (private Qs)", True), ("COMPLIANCE (public Qs)", False)]:
        mask = md["expects_refusal"] if is_private else ~md["expects_refusal"]
        n = int(mask.sum())

        test = _mcnemar_test(
            md_correct=md.loc[mask, "correct"],
            plain_correct=pl.loc[mask, "correct"],
        )

        md_rate = md.loc[mask, "correct"].mean()
        pl_rate = pl.loc[mask, "correct"].mean()
        sig = " *" if test["p_value"] < ALPHA else ""

        print(f"\n  {label} (n={n} paired):")
        print(f"    Markdown: {md_rate:.1%}  |  Plain: {pl_rate:.1%}")
        print(f"    Delta: {(md_rate - pl_rate) * 100:+.1f} pp")
        print(f"    Discordant: {test['n_discordant']}")
        print(f"      b (MD correct, PL wrong): {test['b']}")
        print(f"      c (PL correct, MD wrong): {test['c']}")
        print(f"    McNemar's ({test['method']}): p={test['p_value']:.4f}{sig}")


def _cross_model_summary() -> None:
    """Print cross-model comparison table."""
    print(f"\n{'=' * 60}")
    print("  Cross-Model Summary (* = p < 0.05)")
    print(f"{'=' * 60}")
    print(
        f"\n  {'Model':<8s} {'MD Ref':>8s} {'PL Ref':>8s} {'D(pp)':>7s} {'p(ref)':>10s}"
        f"  {'MD Comp':>8s} {'PL Comp':>8s} {'D(pp)':>7s} {'p(comp)':>10s}"
    )
    print("  " + "-" * 80)

    for model_size in MODEL_SIZES:
        path = f"{RESULTS_DIR}/{model_size}/results.csv"
        df = pd.read_csv(path)
        df["correct"] = df["expects_refusal"] == df["model_refused"]

        md = df[df["prompt_format"] == "markdown"].set_index("question_id").sort_index()
        pl = df[df["prompt_format"] == "plain"].set_index("question_id").sort_index()

        priv = md["expects_refusal"]
        pub = ~md["expects_refusal"]

        ref = _mcnemar_test(
            md_correct=md.loc[priv, "correct"],
            plain_correct=pl.loc[priv, "correct"],
        )
        comp = _mcnemar_test(
            md_correct=md.loc[pub, "correct"],
            plain_correct=pl.loc[pub, "correct"],
        )

        md_ref = md.loc[priv, "correct"].mean()
        pl_ref = pl.loc[priv, "correct"].mean()
        md_comp = md.loc[pub, "correct"].mean()
        pl_comp = pl.loc[pub, "correct"].mean()

        sig_r = "*" if ref["p_value"] < ALPHA else ""
        sig_c = "*" if comp["p_value"] < ALPHA else ""

        print(
            f"  {model_size.upper():<8s}"
            f" {md_ref:>7.1%} {pl_ref:>8.1%} {(md_ref - pl_ref) * 100:>+6.1f}"
            f"  {ref['p_value']:>8.4f}{sig_r:<2s}"
            f" {md_comp:>8.1%} {pl_comp:>8.1%} {(md_comp - pl_comp) * 100:>+6.1f}"
            f"  {comp['p_value']:>8.4f}{sig_c:<2s}"
        )


def main() -> None:
    """Run behavioral McNemar's analysis for all models."""
    print("BEHAVIORAL McNEMAR'S TESTS — MD vs Plain")
    print("Primary: Refusal rate (private Qs)")
    print("Secondary: Compliance rate (public Qs)")

    for model_size in MODEL_SIZES:
        _run_model(model_size=model_size)

    _cross_model_summary()


if __name__ == "__main__":
    main()
