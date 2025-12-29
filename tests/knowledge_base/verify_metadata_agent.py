from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from knowledge_base.config.settings import Settings
from utils.logging import logger

from .verify_utils import get_project_root, load_all_chunks, setup_verification_env


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


class MetadataAuditor:
    """Consolidated auditor for verifying chunk metadata using LangChain agents."""

    def __init__(self, settings: Settings, max_workers: int = 10) -> None:
        self.settings = settings
        self.max_workers = max_workers
        self.project_root = get_project_root()
        self.model = self._init_model()
        self.agent = self._init_agent()

    def _init_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            model="openai/gpt-oss-120b",
            openai_api_key=self.settings.openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0,
        )

    def _init_agent(self) -> Any:  # noqa: ANN401
        @tool
        def get_file_snippet(path: str, start_line: int, end_line: int) -> str:
            """Reads a text snippet from a source file given its path and 1-indexed line range."""
            try:
                file_path = Path(path)
                if not file_path.exists():
                    file_path = self.project_root / path

                if not file_path.exists():
                    return f"Error: File not found at {path}"

                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                snippet_lines = lines[max(0, start_line - 1) : end_line]
                return "".join(snippet_lines)
            except Exception as e:
                return f"Error reading file: {e}"

        return create_agent(
            model=self.model,
            tools=[get_file_snippet],
            response_format=ToolStrategy(AuditResult),
            system_prompt=(
                "You are an expert Knowledge Base Auditor. Verify if chunk metadata "
                "(lines, pages, privacy) is correct.\n"
                "1. For MD files: Use `get_file_snippet` to verify text match.\n"
                "2. For PDFs: Use provided page numbers. Do NOT use tool on PDFs.\n"
                "3. Privacy: public (papers/recipes), private (secrets/salaries), mixed (both)."
            ),
        )

    def audit_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        """Audits a single chunk using the initialized agent."""
        chunk_id = chunk.get("chunk_id", "unknown")
        source = chunk.get("source_file", "unknown")
        s_line = chunk.get("start_line")
        e_line = chunk.get("end_line")
        s_page = chunk.get("start_page")
        e_page = chunk.get("end_page")
        privacy = chunk.get("privacy_level", "unknown")

        if s_line is None and s_page is None:
            return {"chunk_id": chunk_id, "status": "MISSING_METADATA", "source": source}

        try:
            response = self.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"Source: {source}\nLines: {s_line}-{e_line}\n"
                                f"Pages: {s_page}-{e_page}\nLabel: {privacy}\n"
                                f"Text: {chunk.get('document')}"
                            ),
                        }
                    ]
                }
            )
            audit: AuditResult = response["structured_response"]
            return {
                "chunk_id": chunk_id,
                "source": source,
                "lines": f"{s_line}-{e_line}",
                "pages": f"{s_page}-{e_page}",
                "original_label": privacy,
                "text_match": audit.text_match,
                "privacy_match": audit.privacy_match,
                "suggested": audit.suggested_privacy_label,
                "reasoning": audit.reasoning,
                "status": "SUCCESS",
            }
        except Exception as e:
            return {"chunk_id": chunk_id, "status": "ERROR", "reasoning": str(e), "source": source}

    def run_full_audit(self, chunks: list[dict[str, Any]]) -> pd.DataFrame:
        """Runs the audit in parallel over a list of chunks."""
        logger.info(f"Starting parallel audit for {len(chunks)} chunks...")

        audit_records = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # We wrap chunks and audit_chunk to track progress with tqdm
            results = list(tqdm(executor.map(self.audit_chunk, chunks), total=len(chunks)))
            audit_records.extend(results)

        df = pd.DataFrame(audit_records)
        return self._post_process_report(df)

    def _post_process_report(self, df: pd.DataFrame) -> pd.DataFrame:
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
        report_path = self.project_root / "tests" / "audit_results_1.0.csv"
        df.to_csv(report_path, index=False)
        logger.info(f"Audit report saved to {report_path}")
        return df


if __name__ == "__main__":
    # Setup environment
    settings, store = setup_verification_env(log_file="verify_agent.log")

    # Load data
    chunks = load_all_chunks(store)

    if not chunks:
        logger.warning("No chunks found. Is ingestion complete?")
    else:
        auditor = MetadataAuditor(settings)
        report = auditor.run_full_audit(chunks)

        # Summary statistics
        success_df = report[report["status"] == "SUCCESS"]
        if not success_df.empty:
            logger.success(
                f"Audit complete. Text Match: {success_df['text_match'].mean() * 100:.1f}%, "
                f"Privacy Match: {success_df['privacy_match'].mean() * 100:.1f}%"
            )
        else:
            logger.error("No successful audits to report.")
