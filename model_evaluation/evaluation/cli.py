"""CLI entry point for the evaluation pipeline."""

from pathlib import Path

import click

from model_evaluation.evaluation.runner import run_evaluation
from utils.logging import setup_logging

DEFAULT_QUESTIONS_PATH = Path("data_generation/questions")
DEFAULT_OUTPUT_DIR = Path("results")
VALID_MODEL_SIZES = ("1b", "4b", "12b", "27b")


@click.command()
@click.option(
    "--model-size",
    type=click.Choice(VALID_MODEL_SIZES),
    required=True,
    help="Gemma model size to evaluate.",
)
@click.option(
    "--questions",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_QUESTIONS_PATH,
    show_default=True,
    help="Path to a questions CSV file or directory of CSVs.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory for evaluation output files.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from previous run, skipping completed questions.",
)
@click.option(
    "--check-groundedness/--no-check-groundedness",
    default=True,
    show_default=True,
    help="Enable inline groundedness checks with retry on hallucination.",
)
@click.option(
    "--max-groundedness-retries",
    type=int,
    default=2,
    show_default=True,
    help="Maximum retry attempts for hallucinated responses.",
)
@click.option(
    "--rerun-hallucinated",
    is_flag=True,
    default=False,
    help="Remove hallucinated rows from results and re-run them. Implies --resume.",
)
def main(
    *,
    model_size: str,
    questions: Path,
    output_dir: Path,
    resume: bool,
    check_groundedness: bool,
    max_groundedness_retries: int,
    rerun_hallucinated: bool,
) -> None:
    """Run safety prompt evaluation comparing markdown vs plain formats."""
    setup_logging(level="INFO", log_file="evaluation.log")

    if rerun_hallucinated:
        from model_evaluation.evaluation.runner import remove_hallucinated_rows

        results_path = output_dir / model_size / "results.csv"
        remove_hallucinated_rows(results_path=results_path)
        resume = True

    run_evaluation(
        model_size=model_size,
        questions_path=questions,
        output_dir=output_dir,
        resume=resume,
        enable_groundedness=check_groundedness,
        max_groundedness_retries=max_groundedness_retries,
    )


if __name__ == "__main__":
    main()
