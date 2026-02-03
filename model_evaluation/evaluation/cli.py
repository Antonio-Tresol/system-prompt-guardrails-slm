"""CLI entry point for the evaluation pipeline."""

from pathlib import Path

import click

from model_evaluation.evaluation.runner import run_evaluation

DEFAULT_QUESTIONS_PATH = Path("model_evaluation/questions/synthetic_questions.csv")
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
    help="Path to questions CSV file.",
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
def main(
    *,
    model_size: str,
    questions: Path,
    output_dir: Path,
    resume: bool,
) -> None:
    """Run safety prompt evaluation comparing markdown vs plain formats."""
    run_evaluation(
        model_size=model_size,
        questions_path=questions,
        output_dir=output_dir,
        resume=resume,
    )


if __name__ == "__main__":
    main()
