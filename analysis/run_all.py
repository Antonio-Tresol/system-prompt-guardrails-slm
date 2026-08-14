"""Run all analysis scripts in sequence.

Produces a complete, reproducible statistical report of all results.

Usage:
    uv run python -m analysis.run_all
"""

from analysis.behavioral_mcnemar import main as behavioral
from analysis.effective_refusal import main as effective
from analysis.groundedness import main as groundedness
from analysis.leakage_mcnemar import main as leakage_mcnemar
from analysis.leakage_summary import main as leakage_summary
from analysis.per_universe import main as per_universe
from analysis.trajectory_stats import main as trajectory


def main() -> None:
    """Run all analysis scripts."""
    print("=" * 70)
    print("  COMPLETE ANALYSIS REPORT")
    print("  Reproducible statistical analysis from results CSVs")
    print("=" * 70)

    sections = [
        ("1. BEHAVIORAL McNEMAR'S TESTS", behavioral),
        ("2. PER-UNIVERSE BREAKDOWN", per_universe),
        ("3. GROUNDEDNESS ANALYSIS", groundedness),
        ("4. TRAJECTORY & EFFICIENCY", trajectory),
        ("5. LEAKAGE SUMMARY", leakage_summary),
        ("6. LEAKAGE McNEMAR'S TEST", leakage_mcnemar),
        ("7. EFFECTIVE REFUSAL RATE", effective),
    ]

    for title, func in sections:
        print(f"\n\n{'#' * 70}")
        print(f"# {title}")
        print(f"{'#' * 70}\n")
        func()

    print(f"\n\n{'=' * 70}")
    print("  REPORT COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
