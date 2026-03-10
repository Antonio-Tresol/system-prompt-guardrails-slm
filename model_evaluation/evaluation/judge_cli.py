"""CLI entry point for standalone judge backfill on existing results."""

from pathlib import Path

import click

from model_evaluation.evaluation.judge import backfill_groundedness, backfill_results
from utils.logging import setup_logging


@click.command()
@click.option(
    "--results",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to existing results CSV file to backfill with judge classifications.",
)
@click.option(
    "--groundedness",
    is_flag=True,
    default=False,
    help="Backfill groundedness instead of refusal classifications.",
)
def main(*, results: Path, groundedness: bool) -> None:
    """Backfill judge classifications on an existing results CSV."""
    setup_logging(level="INFO", log_file="judge_backfill.log")
    if groundedness:
        backfill_groundedness(results_path=results)
    else:
        backfill_results(results_path=results)


if __name__ == "__main__":
    main()
