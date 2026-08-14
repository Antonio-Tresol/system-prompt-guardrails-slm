"""Per-universe breakdown of refusal and compliance rates (exploratory).

Runs McNemar's test per universe context, broken down by model size.
Underpowered for small effects (75 paired questions per universe).

Usage:
    uv run python -m analysis.per_universe
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
        method = "exact"
    else:
        stat = (b - c) ** 2 / n_discordant
        p_value = 1 - chi2.cdf(stat, df=1)
        method = "chi2"

    return {"b": b, "c": c, "n_discordant": n_discordant, "p_value": p_value, "method": method}


def _run_model(*, model_size: str) -> None:
    """Run per-universe McNemar's for one model."""
    path = f"{RESULTS_DIR}/{model_size}/results.csv"
    df = pd.read_csv(path)
    df["correct"] = df["expects_refusal"] == df["model_refused"]

    md = df[df["prompt_format"] == "markdown"].set_index("question_id").sort_index()
    pl = df[df["prompt_format"] == "plain"].set_index("question_id").sort_index()
    universes = sorted(df["universe_context"].unique())

    print(f"\n{'=' * 60}")
    print(f"  {model_size.upper()} MODEL — Per-Universe Breakdown (exploratory)")
    print(f"{'=' * 60}")

    for label, is_private in [("REFUSAL (private)", True), ("COMPLIANCE (public)", False)]:
        print(f"\n  {label}:")
        header = (
            f"  {'Universe':<30s} {'MD':>7s} {'Plain':>7s}"
            f" {'D(pp)':>7s}  {'b':>3s}/{'c':>3s}  {'p':>8s}"
        )
        print(header)
        print("  " + "-" * 72)

        for ctx in universes:
            ctx_mask = md["universe_context"] == ctx
            if is_private:
                mask = ctx_mask & md["expects_refusal"]
            else:
                mask = ctx_mask & ~md["expects_refusal"]

            n = int(mask.sum())
            if n == 0:
                continue

            test = _mcnemar_test(
                md_correct=md.loc[mask, "correct"],
                plain_correct=pl.loc[mask, "correct"],
            )
            md_rate = md.loc[mask, "correct"].mean()
            pl_rate = pl.loc[mask, "correct"].mean()
            sig = "*" if test["p_value"] < ALPHA else ""

            display_name = ctx.replace("_", " ").title()[:28]
            print(
                f"  {display_name:<30s}"
                f" {md_rate:>6.1%} {pl_rate:>7.1%} {(md_rate - pl_rate) * 100:>+6.1f}"
                f"  {test['b']:>3d}/{test['c']:>3d}"
                f"  {test['p_value']:>7.4f}{sig}"
            )


def main() -> None:
    """Run per-universe analysis for all models."""
    print("PER-UNIVERSE BREAKDOWN — Refusal & Compliance by Domain (exploratory)")
    print("75 paired questions per universe — underpowered for small effects")

    for model_size in MODEL_SIZES:
        _run_model(model_size=model_size)


if __name__ == "__main__":
    main()
