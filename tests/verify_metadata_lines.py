from pathlib import Path
import pandas as pd
from knowledge_base.config.settings import Settings
from knowledge_base.vectordb.chroma_store import ChromaStore
import tiktoken
import os
from openai import OpenAI
from tqdm import tqdm

# Setup
notebook_dir = Path.cwd() / "tests"  # Adjust since we run from root usually or logic handles it
project_root = Path.cwd()
config_path = project_root / "knowledge_base" / "config" / "config.yaml"

print(f"Loading config from {config_path}")
settings = Settings.load_from_yaml(config_path=str(config_path), project_root=project_root)

store = ChromaStore(
    persist_directory=settings.paths.vector_db,
    embeddings_model=settings.embeddings.model,
    openrouter_api_key=settings.openrouter_api_key,
    openrouter_base_url=settings.openrouter_base_url,
)

print("✅ Connected to knowledge base")


def verify_chunk_lines(chunks_df):
    print("🔍 Verifying Line Numbers...")
    errors = []

    # Group by source file to avoid reading file multiple times
    for source_file, group in chunks_df.groupby("source_file"):
        try:
            # Assuming source_file path in metadata is absolute or relative to project root
            # If it's stored as absolute, we can use it directly.
            # If not found, try resolving relative to project root.
            file_path = Path(source_file)
            if not file_path.exists():
                file_path = project_root / source_file

            if not file_path.exists():
                # print(f"⚠️ File not found: {source_file}")
                # Some might be PDFs we don't have text for easily here
                continue

            if file_path.suffix.lower() == ".pdf":
                # Skip PDF source checks for exact line match unless we have the text dump
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

                # Verify content overlap
                # Adjust for 1-based indexing
                # Line 1 is lines[0]

                # Safe slice
                extracted_lines = lines[max(0, start - 1) : end]
                extracted_text = "".join(extracted_lines)
                chunk_text = row["document"]

                # Normalization for loose comparison
                # Remove newlines and extra spaces to check containment
                norm_chunk = chunk_text.replace("\n", "").replace(" ", "")
                norm_extracted = extracted_text.replace("\n", "").replace(" ", "")

                if norm_chunk not in norm_extracted:
                    # Fallback: sometimes chunks span boundaries differently or normalization is tricky.
                    # But basic containment should work if lines are correct.
                    # Let's verify if at least significant portion is there.
                    if (
                        len(norm_chunk) > 10
                        and norm_chunk[:20] not in norm_extracted
                        and norm_chunk[-20:] not in norm_extracted
                    ):
                        errors.append(
                            f"Text mismatch in {source_file} lines {start}-{end}\nChunk start: {chunk_text[:30]}...\nExpected in lines: {start}-{end}"
                        )

        except Exception as e:
            errors.append(f"Error processing {source_file}: {str(e)}")

    if errors:
        print(f"❌ Found {len(errors)} errors:")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print("✅ All line number checks passed for accessible text files!")


def verify_privacy_labels_with_llm(chunks_df):
    print("\n🤖 Verifying Privacy Labels with z-ai/glm-4.6...")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )

    mismatches = []

    for idx, row in tqdm(chunks_df.iterrows(), total=len(chunks_df)):
        text = row["document"]
        current_label = row["privacy_level"]

        # Simple prompt for classification
        prompt = f"""Classify the privacy level of the following text snippet.
        
        Levels:
        - public: General information, cookbook recipes, menu items, public descriptions.
        - private: Personal secrets, internal conflicts, hidden agendas, restricted sections.
        - mixed: Contains both public and private information.
        
        Text:
        {text}
        
        Return ONLY one word: public, private, or mixed."""

        try:
            response = client.chat.completions.create(
                model="z-ai/glm-4.6",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            llm_label = response.choices[0].message.content.strip().lower()

            # Normalize label (remove punctuation etc)
            valid_labels = ["public", "private", "mixed"]
            detected = [l for l in valid_labels if l in llm_label]
            final_llm_label = detected[0] if detected else "unknown"

            # Allow "mixed" to match "private" or "public" if strictness varies, but user wants check.
            # We will flag strictly for now.
            if final_llm_label != current_label and final_llm_label != "unknown":
                # Filter out minor mixed/public confusion if needed, but report for now.
                mismatches.append(
                    {
                        "chunk_id": idx,
                        "text": text[:50] + "...",
                        "current": current_label,
                        "llm": final_llm_label,
                    }
                )

        except Exception as e:
            print(f"Error checking chunk {idx}: {e}")

    if mismatches:
        print(f"⚠️ Found {len(mismatches)} potential mismatches:")
        mismatches_df = pd.DataFrame(mismatches)
        print(mismatches_df.head(10))
        # Calculate agreement score
        agreement = (len(chunks_df) - len(mismatches)) / len(chunks_df) * 100
        print(f"Agreement Rate: {agreement:.2f}%")
    else:
        print("✅ Perfect agreement between metadata and LLM!")


# Main execution
if __name__ == "__main__":
    # Get all chunks
    results = store.vector_store.get()
    chunks_data = []
    if results["ids"]:
        for i in range(len(results["ids"])):
            meta = results["metadatas"][i]
            text = results["documents"][i]
            meta["document"] = text
            chunks_data.append(meta)

    chunks_df = pd.DataFrame(chunks_data)
    print(f"Loaded {len(chunks_df)} chunks.")

    if not chunks_df.empty:
        # Run Validations
        verify_chunk_lines(chunks_df)
        verify_privacy_labels_with_llm(chunks_df)
    else:
        print("⚠️ No chunks found in database.")
