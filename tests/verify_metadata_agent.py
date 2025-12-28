import pandas as pd
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from knowledge_base.config.settings import Settings
from knowledge_base.vectordb.chroma_store import ChromaStore

# Setup
PROJECT_ROOT = Path.cwd()
CONFIG_PATH = PROJECT_ROOT / "knowledge_base" / "config" / "config.yaml"

settings = Settings.load_from_yaml(config_path=str(CONFIG_PATH), project_root=PROJECT_ROOT)

# Initialize ChromaStore
store = ChromaStore(
    persist_directory=settings.paths.vector_db,
    embeddings_model=settings.embeddings.model,
    openrouter_api_key=settings.openrouter_api_key,
    openrouter_base_url=settings.openrouter_base_url,
)


# 1. Structured Output Schema
class AuditResult(BaseModel):
    """Schema for the metadata audit result."""

    text_match: bool = Field(
        description="True if the chunk text is correctly found in the specified file lines."
    )
    privacy_match: bool = Field(
        description="True if the current privacy label accurately describes the content."
    )
    suggested_privacy_label: Literal["public", "private", "mixed"] = Field(
        description="The most appropriate privacy label."
    )
    reasoning: str = Field(description="Explanation for the match status and label choice.")


# 2. Tools
@tool
def get_file_snippet(
    path: str,
    start_line: int,
    end_line: int,
) -> str:
    """Reads a text snippet from a source file given its path and 1-indexed line range.

    Args:
        path: Path to the file.
        start_line: The first line to read (1-indexed).
        end_line: The last line to read (1-indexed, inclusive).
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            file_path = PROJECT_ROOT / path

        if not file_path.exists():
            return f"Error: File not found at {path}"

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 1-indexed to 0-indexed slicing
        snippet_lines = lines[max(0, start_line - 1) : end_line]
        return "".join(snippet_lines)
    except Exception as e:
        return f"Error reading file: {e}"


# 3. Model & Agent
# Using modern create_agent with ToolStrategy for structured output
model = ChatOpenAI(
    model="moonshotai/kimi-k2-thinking",
    openai_api_key=settings.openrouter_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)

agent = create_agent(
    model=model,
    tools=[get_file_snippet],
    response_format=ToolStrategy(AuditResult),
    system_prompt=(
        "You are an expert Knowledge Base Auditor. Your goal is to verify if a chunk's "
        "metadata (line numbers, page numbers, and privacy label) is correct.\n\n"
        "GUIDELINES:\n"
        "1. For Markdown (.md) files: Use the `get_file_snippet` tool to pull the exact lines "
        "specified. The text SHOULD match. If it doesn't, mark `text_match=False`.\n"
        "2. For PDF files: Page numbers are provided. You CANNOT use `get_file_snippet` on PDFs "
        "(it will return error/binary). FOCUS on verifying if the privacy label is appropriate "
        "given the content.\n"
        "3. Privacy Labels definitions:\n"
        "   - public: General info, public records, academic papers, cookbook recipes.\n"
        "   - private: Personal secrets, internal agendas, hidden data, salaries.\n"
        "   - mixed: Both public and private elements."
    ),
)


def audit_one_chunk(i: int, results: dict, audit_records: list) -> None:
    """Audits a single chunk and appends to audit_records."""
    meta = results["metadatas"][i]
    text = results["documents"][i]
    chunk_id = results["ids"][i]

    s_line = meta.get("start_line")
    e_line = meta.get("end_line")
    s_page = meta.get("start_page")
    e_page = meta.get("end_page")
    source = meta.get("source_file", "unknown")

    if s_line is None and s_page is None:
        audit_records.append(
            {
                "chunk_id": chunk_id,
                "status": "MISSING_METADATA",
                "source": source,
            }
        )
        return

    try:
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Audit this chunk:\n"
                            f"Source: {source}\n"
                            f"Lines: {s_line} to {e_line}\n"
                            f"Pages: {s_page} to {e_page}\n"
                            f"Current Label: {meta.get('privacy_level')}\n"
                            f"Chunk Text: {text}"
                        ),
                    }
                ]
            }
        )

        # Extract structured response
        audit: AuditResult = response["structured_response"]

        audit_records.append(
            {
                "chunk_id": chunk_id,
                "source": source,
                "lines": f"{s_line}-{e_line}",
                "pages": f"{s_page}-{e_page}",
                "original_label": meta.get("privacy_level"),
                "text_match": audit.text_match,
                "privacy_match": audit.privacy_match,
                "suggested": audit.suggested_privacy_label,
                "reasoning": audit.reasoning,
                "status": "SUCCESS",
            }
        )

    except Exception as e:
        audit_records.append(
            {
                "chunk_id": chunk_id,
                "status": "ERROR",
                "reasoning": str(e),
                "source": source,
            }
        )


def run_audit() -> None:
    print("🚀 Starting modern LangChain agent audit (Parallel)...")
    results = store.vector_store.get()

    if not results["ids"]:
        print("⚠️ No chunks found. Is the ingestion complete?")
        return

    audit_records = []
    num_chunks = len(results["ids"])

    from concurrent.futures import ThreadPoolExecutor

    # Audit all chunks in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(
            tqdm(
                executor.map(
                    lambda i: audit_one_chunk(i, results, audit_records), range(num_chunks)
                ),
                total=num_chunks,
            )
        )

    # Report
    df = pd.DataFrame(audit_records)
    # Ensure columns exist even if some chunks failed
    cols = [
        "chunk_id",
        "source",
        "lines",
        "pages",
        "original_label",
        "text_match",
        "privacy_match",
        "suggested",
        "reasoning",
        "status",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = None

    df = df[cols]
    df.to_csv(PROJECT_ROOT / "tests" / "audit_results_1.0.csv", index=False)

    print(f"\n✅ Audit complete for {len(df)} chunks.")
    success_df = df[df["status"] == "SUCCESS"]
    if not success_df.empty:
        print(f"📊 Text Match Rate: {(success_df['text_match'].mean() * 100):.1f}%")
        print(f"📊 Privacy Match Rate: {(success_df['privacy_match'].mean() * 100):.1f}%")
    else:
        print("📊 No successful audits to report rates.")


if __name__ == "__main__":
    run_audit()
