"""CLI for question generation.

Usage:
    uv run generate_questions
    uv run generate_questions --dry-run
    uv run generate_questions --num-refusal 30 --num-non-refusal 30
"""

import csv
from pathlib import Path

import click

from model_evaluation.config import Settings
from model_evaluation.question_generator.agent import (
    GeneratedQuestion,
    generate_questions,
)

# Output path for generated questions
QUESTIONS_CSV_PATH = Path(__file__).parent.parent / "questions" / "synthetic_questions.csv"


def load_existing_questions() -> list[str]:
    """Load existing questions from CSV to avoid duplicates.

    Returns:
        List of existing question texts.
    """
    if not QUESTIONS_CSV_PATH.exists():
        return []

    questions = []
    with QUESTIONS_CSV_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row.get("Question", ""))
    return [q for q in questions if q]


def save_questions(questions: list[GeneratedQuestion]) -> None:
    """Save generated questions to CSV.

    Args:
        questions: List of generated questions to save.
    """
    # Ensure directory exists
    QUESTIONS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with QUESTIONS_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["Number", "Question", "Universe Context", "Is Refusal"]
        writer = csv.DictWriter(fieldnames=fieldnames, f=f)
        writer.writeheader()

        for idx, q in enumerate(questions, 1):
            writer.writerow(
                {
                    "Number": idx,
                    "Question": q.question,
                    "Universe Context": q.universe_context,
                    "Is Refusal": "Yes" if q.is_refusal else "No",
                }
            )


@click.command()
@click.option(
    "--num-refusal",
    default=30,
    help="Number of refusal questions to generate",
)
@click.option(
    "--num-non-refusal",
    default=30,
    help="Number of non-refusal questions to generate",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be generated without saving",
)
@click.option(
    "--append",
    is_flag=True,
    help="Append to existing questions instead of replacing",
)
def main(
    *,
    num_refusal: int,
    num_non_refusal: int,
    dry_run: bool,
    append: bool,
) -> None:
    """Generate synthetic test questions for safety evaluation."""
    click.echo("🚀 Starting question generation...")
    click.echo(f"   Refusal questions: {num_refusal}")
    click.echo(f"   Non-refusal questions: {num_non_refusal}")
    click.echo(f"   Total: {num_refusal + num_non_refusal}")
    click.echo()

    # Load existing questions if appending
    existing_questions: list[str] = []
    if append and QUESTIONS_CSV_PATH.exists():
        existing_questions = load_existing_questions()
        click.echo(f"📚 Found {len(existing_questions)} existing questions")

    # Load settings
    settings = Settings()  # type: ignore[call-arg]

    click.echo("🤖 Generating questions with Deep Agent...")
    questions = generate_questions(
        num_refusal=num_refusal,
        num_non_refusal=num_non_refusal,
        existing_questions=existing_questions,
        settings=settings,
    )

    click.echo(f"\n✅ Generated {len(questions)} questions:")

    # Count by type
    refusal_count = sum(1 for q in questions if q.is_refusal)
    non_refusal_count = len(questions) - refusal_count
    click.echo(f"   Refusal: {refusal_count}")
    click.echo(f"   Non-refusal: {non_refusal_count}")

    # Count by universe
    universe_counts: dict[str, int] = {}
    for q in questions:
        universe_counts[q.universe_context] = universe_counts.get(q.universe_context, 0) + 1
    click.echo("\n📊 Distribution by universe:")
    for universe, count in sorted(universe_counts.items()):
        click.echo(f"   {universe}: {count}")

    if dry_run:
        click.echo("\n🔍 Dry run - showing first 10 questions:")
        for q in questions[:10]:
            refusal_marker = "🔴" if q.is_refusal else "🟢"
            click.echo(f"   {refusal_marker} [{q.universe_context}] {q.question[:60]}...")
        click.echo("\n⚠️  Dry run mode - no file was written")
    else:
        save_questions(questions)
        click.echo(f"\n💾 Saved to: {QUESTIONS_CSV_PATH}")


if __name__ == "__main__":
    main()
