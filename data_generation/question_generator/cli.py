"""CLI for generating evaluation questions.

This module provides a command-line interface for generating evaluation
questions grounded in the synthetic cookbook documents.

Usage:
    uv run generate-questions --document carnelian_table --count 30
    uv run generate-questions --document all --count 60 --type both
"""

import argparse
import asyncio
import sys
from pathlib import Path

from data_generation.question_generator.agent import generate_questions
from data_generation.question_generator.schemas import DOCUMENT_YAML_MAP


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate evaluation questions for safety testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run generate-questions --document carnelian_table --count 30
  uv run generate-questions --document all --count 60 --type private
  uv run generate-questions --document brine_and_riddle --count 20 --output custom.csv
        """,
    )

    parser.add_argument(
        "--document",
        "-d",
        type=str,
        required=True,
        choices=list(DOCUMENT_YAML_MAP.keys()) + ["all"],
        help="Document to generate questions for, or 'all' for all documents",
    )

    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=30,
        help="Number of questions to generate per document (default: 30)",
    )

    parser.add_argument(
        "--type",
        "-t",
        type=str,
        default="both",
        choices=["private", "public", "both"],
        help="Type of questions to generate (default: both)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output CSV file path (default: model_evaluation/questions/generated_questions.csv)",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to CSV, just print results",
    )

    return parser.parse_args()


async def run_generation(args: argparse.Namespace) -> None:
    """Run the question generation based on parsed arguments.

    Args:
        args: Parsed command line arguments.
    """
    # Determine output path
    if args.no_save:
        output_path = None
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("model_evaluation/questions/generated_questions.csv")

    # Determine which documents to process
    documents = list(DOCUMENT_YAML_MAP.keys()) if args.document == "all" else [args.document]

    total_questions: list[dict[str, str]] = []

    for doc in documents:
        print(f"\n{'=' * 60}")
        print(f"Generating {args.count} {args.type} questions for: {doc}")
        print(f"{'=' * 60}")

        questions = await generate_questions(
            document_name=doc,
            count=args.count,
            question_type=args.type,
            output_path=output_path,
        )

        total_questions.extend(questions)

        # Print summary for this document
        private_count = sum(1 for q in questions if q["Malicious question"] == "Yes")
        public_count = len(questions) - private_count
        print(f"\nGenerated {len(questions)} questions:")
        print(f"  - Private (expects refusal): {private_count}")
        print(f"  - Public (expects answer): {public_count}")

        # Print a few examples
        print("\nSample questions:")
        for q in questions[:3]:
            q_type = "PRIVATE" if q["Malicious question"] == "Yes" else "PUBLIC"
            print(f"  [{q_type}] {q['Question'][:80]}...")

    # Final summary
    print(f"\n{'=' * 60}")
    print("GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total questions generated: {len(total_questions)}")

    if output_path:
        print(f"Saved to: {output_path}")
    else:
        print("(Not saved - --no-save flag was used)")


def main() -> None:
    """Entry point for the CLI."""
    args = parse_args()

    try:
        asyncio.run(run_generation(args))
    except KeyboardInterrupt:
        print("\n\nGeneration cancelled by user.")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
