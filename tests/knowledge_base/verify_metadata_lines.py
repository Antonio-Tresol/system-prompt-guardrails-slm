from pathlib import Path

import pandas as pd

from utils.logging import logger

from .verify_utils import get_project_root, load_all_chunks, setup_verification_env


def verify_chunk_lines(chunks_df: pd.DataFrame) -> None:
    """Verifies chunk text match against source file line ranges."""
    logger.info("Verifying Line Numbers...")
    errors = []
    project_root = get_project_root()

    # Group by source file to avoid reading file multiple times
    for source_file, group in chunks_df.groupby("source_file"):
        try:
            file_path = Path(source_file)
            if not file_path.exists():
                file_path = project_root / source_file

            if not file_path.exists():
                # logger.debug(f"File not found: {source_file}")
                continue

            if file_path.suffix.lower() == ".pdf":
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for _, row in group.iterrows():
                start = row.get("start_line")
                end = row.get("end_line")

                if pd.isna(start) or pd.isna(end):
                    errors.append(f"Missing line numbers in {source_file}")
                    continue

                start, end = int(start), int(end)

                if start > end:
                    errors.append(f"Invalid range {start}-{end} in {source_file}")
                    continue

                # Adjust for 1-based indexing: Line 1 is lines[0]
                extracted_lines = lines[max(0, start - 1) : end]
                extracted_text = "".join(extracted_lines)
                chunk_text = row["document"]

                # Normalization for containment check
                norm_chunk = chunk_text.replace("\n", "").replace(" ", "")
                norm_extracted = extracted_text.replace("\n", "").replace(" ", "")

                if norm_chunk not in norm_extracted and (
                    len(norm_chunk) > 10
                    and norm_chunk[:20] not in norm_extracted
                    and norm_chunk[-20:] not in norm_extracted
                ):
                    err_msg = (
                        f"Text mismatch in {source_file} lines {start}-{end}\n"
                        f"Chunk (starts with): {chunk_text[:30]}..."
                    )
                    errors.append(err_msg)

        except Exception as e:
            errors.append(f"Error processing {source_file}: {str(e)}")

    if errors:
        logger.error(f"Found {len(errors)} line mapping errors:")
        for e in errors[:10]:
            logger.error(f"  - {e}")
        if len(errors) > 10:
            logger.error(f"  ... and {len(errors) - 10} more")
    else:
        logger.success("All line number checks passed for accessible text files!")


if __name__ == "__main__":
    # Setup environment
    settings, store = setup_verification_env(log_file="verify_lines.log")

    # Load data
    chunks_data = load_all_chunks(store)
    chunks_df = pd.DataFrame(chunks_data)

    if not chunks_df.empty:
        verify_chunk_lines(chunks_df)
    else:
        logger.warning("No chunks found in database to verify.")
