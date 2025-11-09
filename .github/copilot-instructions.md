# GitHub Copilot Instructions

## Quality Standards

Always adhere to the project's quality standards. Before finalizing your work, you **must** run the following commands in the terminal and ensure they all pass without errors:

### 1. Auto-fix Lint Errors
```bash
uv run ruff check --fix .
```

### 2. Auto-format and Sort Imports
```bash
uv run ruff format .
```

### 3. Check Types
```bash
uv run pyrefly check
```

> **Important**: If `pyrefly check` reports any errors, you must fix them before finalizing your work.

---

## Code Style Guidelines

### Type Hints & Docstrings
- All new code **must** be fully type-hinted
- Follow the **Google-style docstring convention** as defined in `pyproject.toml`

### Trailing Commas
- **ALWAYS** include trailing commas in multi-line collections (lists, tuples, dictionaries, function parameters, etc.) to improve readability and make version control diffs cleaner.

### Named Arguments
- Always use **named arguments** when calling functions or methods with multiple parameters
- Enforce this at the function/method definition level using `*` to require named arguments. *NO PARAMETER SHOULD BE POSITIONAL IF THERE ARE MULTIPLE PARAMETERS*.


### Code Comments
- **AVOID** comments that are obvious or redundant
- Only add comments when they provide additional context or clarification that is not immediately clear from the code itself
- Code should be clean, self-explanatory, and follow **SOLID principles**

### Terminology
- Prefer to say **"model"** over "LLM" in code and comments, even when referring to large language models

---

## Framework-Specific Guidelines

### LangChain & LangGraph
This project uses **LangChain** and **LangGraph** heavily.

- **ALWAYS** use the LangChain MCP server tools to read the latest documentation when touching anything related to LangChain or LangGraph
- If you receive links, **ALWAYS** use the MCP server tools to read the content let that be with Docs by Lanchain tools or with the fetch webcontent tool.

---

## Testing Standards

### Testing Framework
- Use **pytest** for all tests

### Test Structure
- Follow the **Arrange-Act-Assert** pattern
- Use **fixtures** where appropriate to set up test data or state
- **Mock external dependencies** to ensure tests are isolated and reliable

### Test Code Quality
Tests should maintain the same quality standards as production code:
- Fully **type-hinted**
- Follow **Google-style docstring convention**
- Be **DRY** (Don't Repeat Yourself)
- Be clean and **self-explanatory**
- Follow **SOLID principles**