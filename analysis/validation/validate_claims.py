"""Validate every reported quantitative claim against the raw data."""

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# Load data
df12 = pd.read_csv("results/12b/results.csv")
df4 = pd.read_csv("results/4b/results.csv")

print("=" * 70)
print("DATA SHAPE CHECK")
print("=" * 70)
print(f"12B: {df12.shape[0]} rows  (expected 1,200)")
print(f"4B:  {df4.shape[0]} rows  (expected 1,200)")

for label, df in [("12B", df12), ("4B", df4)]:
    print(f"\n{'=' * 70}")
    print(f"MODEL: {label}")
    print(f"{'=' * 70}")

    md = df[df["prompt_format"] == "markdown"]
    pt = df[df["prompt_format"] == "plain"]
    print(f"Markdown rows: {len(md)}, Plain rows: {len(pt)}")

    md_priv = md[md["expects_refusal"] == True]
    md_pub = md[md["expects_refusal"] == False]
    pt_priv = pt[pt["expects_refusal"] == True]
    pt_pub = pt[pt["expects_refusal"] == False]
    print(f"MD private: {len(md_priv)}, MD public: {len(md_pub)}")
    print(f"PT private: {len(pt_priv)}, PT public: {len(pt_pub)}")

    md_priv_refusal = md_priv["model_refused"].mean() * 100
    pt_priv_refusal = pt_priv["model_refused"].mean() * 100
    print("\n--- Private Question Refusal Rates ---")
    print(f"Markdown: {md_priv_refusal:.1f}%")
    print(f"Plain:    {pt_priv_refusal:.1f}%")

    md_pub_comply = (md_pub["model_refused"] == False).mean() * 100
    pt_pub_comply = (pt_pub["model_refused"] == False).mean() * 100
    print("\n--- Public Question Compliance Rates ---")
    print(f"Markdown: {md_pub_comply:.1f}%")
    print(f"Plain:    {pt_pub_comply:.1f}%")

    merged_priv = md_priv[["question_id", "model_refused"]].merge(
        pt_priv[["question_id", "model_refused"]], on="question_id", suffixes=("_md", "_pt")
    )
    a = (
        (merged_priv["model_refused_md"] == True) & (merged_priv["model_refused_pt"] == True)
    ).sum()
    b = (
        (merged_priv["model_refused_md"] == True) & (merged_priv["model_refused_pt"] == False)
    ).sum()
    c = (
        (merged_priv["model_refused_md"] == False) & (merged_priv["model_refused_pt"] == True)
    ).sum()
    d = (
        (merged_priv["model_refused_md"] == False) & (merged_priv["model_refused_pt"] == False)
    ).sum()
    print("\n--- McNemar (Private) ---")
    print(f"a={a}, b={b}, c={c}, d={d}, discordant={b + c}")
    table = np.array([[a, b], [c, d]])
    if b + c < 25:
        result = mcnemar(table, exact=True)
        print(f"Exact p = {result.pvalue:.6f}")
    else:
        result = mcnemar(table, exact=False, correction=True)
        print(f"Chi2 p = {result.pvalue:.6f}")

    merged_pub = md_pub[["question_id", "model_refused"]].merge(
        pt_pub[["question_id", "model_refused"]], on="question_id", suffixes=("_md", "_pt")
    )
    a2 = ((merged_pub["model_refused_md"] == True) & (merged_pub["model_refused_pt"] == True)).sum()
    b2 = (
        (merged_pub["model_refused_md"] == True) & (merged_pub["model_refused_pt"] == False)
    ).sum()
    c2 = (
        (merged_pub["model_refused_md"] == False) & (merged_pub["model_refused_pt"] == True)
    ).sum()
    d2 = (
        (merged_pub["model_refused_md"] == False) & (merged_pub["model_refused_pt"] == False)
    ).sum()
    print("\n--- McNemar (Public) ---")
    print(f"a={a2}, b={b2}, c={c2}, d={d2}, discordant={b2 + c2}")
    table2 = np.array([[a2, b2], [c2, d2]])
    if b2 + c2 < 25:
        result2 = mcnemar(table2, exact=True)
        print(f"Exact p = {result2.pvalue:.6f}")
    else:
        result2 = mcnemar(table2, exact=False, correction=True)
        print(f"Chi2 p = {result2.pvalue:.6f}")

    print("\n--- Per-Universe Refusal (Private) ---")
    for uni in sorted(df["universe_context"].unique()):
        md_u = md_priv[md_priv["universe_context"] == uni]
        pt_u = pt_priv[pt_priv["universe_context"] == uni]
        if len(md_u) > 0:
            print(
                f"  {uni}: MD={md_u['model_refused'].mean() * 100:.1f}% PT={pt_u['model_refused'].mean() * 100:.1f}% (n={len(md_u)})"
            )

    merged_all = md[
        ["question_id", "num_steps", "total_input_tokens", "total_output_tokens", "duration_ms"]
    ].merge(
        pt[
            ["question_id", "num_steps", "total_input_tokens", "total_output_tokens", "duration_ms"]
        ],
        on="question_id",
        suffixes=("_md", "_pt"),
    )
    print("\n--- Trajectory ---")
    print(f"Median steps: MD={md['num_steps'].median():.0f}, PT={pt['num_steps'].median():.0f}")
    print(
        f"Median input tokens: MD={md['total_input_tokens'].median():.0f}, PT={pt['total_input_tokens'].median():.0f}"
    )
    print(
        f"Median output tokens: MD={md['total_output_tokens'].median():.0f}, PT={pt['total_output_tokens'].median():.0f}"
    )

    sae_cols = [c for c in df.columns if c.startswith("sae_")]
    if sae_cols:
        print("\n--- SAE ---")
        for c in sae_cols:
            vals = df[c].dropna()
            if len(vals) > 0:
                print(f"  {c}: mean={vals.mean():.4f}")

print(f"\n{'=' * 70}")
print("CONFUSION MATRIX")
print(f"{'=' * 70}")
for label, df in [("12B", df12), ("4B", df4)]:
    tp = ((df["expects_refusal"] == True) & (df["model_refused"] == True)).sum()
    fn = ((df["expects_refusal"] == True) & (df["model_refused"] == False)).sum()
    fp = ((df["expects_refusal"] == False) & (df["model_refused"] == True)).sum()
    tn = ((df["expects_refusal"] == False) & (df["model_refused"] == False)).sum()
    print(f"{label}: TP={tp} FN={fn} FP={fp} TN={tn} Total={tp + fn + fp + tn}")
