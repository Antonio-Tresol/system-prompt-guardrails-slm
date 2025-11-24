"""Script to run knowledge base integration tests with environment variables.

This script loads environment variables from .env file and runs integration tests
that require API keys. Useful for local development.

Usage:
    uv run python scripts/run_integration_tests.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main() -> None:
    """Load environment and run integration tests."""
    import subprocess

    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv

        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            print(f"✅ Loaded environment variables from {env_file}")
        else:
            print(f"⚠️  No .env file found at {env_file}")
            print("   Integration tests will use existing environment variables")
    except ImportError:
        print("⚠️  python-dotenv not installed, using existing environment variables")

    # Run integration tests
    test_path = project_root / "tests" / "knowledge_base" / "test_integration.py"

    print(f"\n🧪 Running integration tests from {test_path}\n")

    result = subprocess.run(  # noqa: S603
        ["uv", "run", "pytest", str(test_path), "-v"],  # noqa: S607
        cwd=str(project_root),
        check=False,
    )

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
