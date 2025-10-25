Always adhere to the project's quality standards. Before finalizing your work, you must run the following commands in the terminal and ensure they all pass without errors:

1.  **Auto-fix Lint Errors:**
    ```bash
    uv run ruff check --fix .
    ```

2.  **Auto-format and Sort Imports:**
    ```bash
    uv run ruff format .
    ```

3.  **Check Types:**
    ```bash
    uv run pyrefly check
    ```

If `pyrefly check` reports any errors, you must fix them. All new code must be fully type-hinted and follow the Google-style docstring convention as defined in `pyproject.toml`.

This project uses langchain and langgraph heavily. As your knowledge may be outdated, ALWAYS use the langchain mcp server tools to read the latest documentation when touching anything related to langchain or langgraph.

AVOID comments on code that are obvious or redundant. Only add comments when they provide additional context or clarification that is not immediately clear from the code itself. The code should be clean and self-explanatory, like solid principles.