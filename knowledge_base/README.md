# Knowledge Base Vector Database

A local ChromaDB-based vector database with rich metadata for RAG applications. Supports Markdown cookbooks and PDF research papers with intelligent privacy detection.

## Features

- **Document Types**: Markdown (`.md`) and PDF (`.pdf`) with OCR support via Docling
- **Privacy Detection**: LLM-based classification of public, mixed, and private content
- **Rich Metadata**: Includes section hierarchy, privacy levels, token counts, character indices
- **Incremental Updates**: Only processes new or modified documents
- **OpenRouter Integration**: Uses OpenRouter for embeddings and LLM privacy detection

## Quick Start

### 1. Configuration

The system loads configuration from:
- Environment variables (`.env` file)
- YAML configuration files

Make sure your `.env` file contains:
```bash
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### 2. Build the Knowledge Base

```bash
uv run build_knowledge_base
```

This will:
1. Load documents from `data_generation/synthetic_data/`
2. Process and chunk them using Docling
3. Detect privacy levels using LLM
4. Extract rich metadata
5. Store in ChromaDB with embeddings
6. Track processed files for incremental updates

### 3. Incremental Updates

Simply run the command again to process new or modified documents:
```bash
uv run build_knowledge_base
```

The system automatically detects changes and only processes updated files.

## Configuration Files

### `config/config.yaml`

Main configuration file:
```yaml
paths:
  source_documents: "./data_generation/synthetic_data"
  vector_db: "./knowledge_base/vectordb/chroma_db"
  pdf_private_config: "./knowledge_base/config/pdf_private_sections.yaml"
  file_tracker: "./knowledge_base/vectordb/file_tracker.json"

embeddings:
  model: "google/gemini-embedding-001"

llm:
  model: "minimax/minimax-m2:free"
  temperature: 0.0

chunking:
  max_chunk_size: 1000
  min_chunk_size: 100

logging:
  level: "INFO"
```

### `config/pdf_private_sections.yaml`

Defines keywords indicating private content in documents.

## Architecture

```
knowledge_base/
├── config/              # Configuration files and settings
├── schemas/             # Pydantic data models
├── ingest/              # Document processing pipeline
│   ├── loaders.py       # Docling document loaders
│   ├── chunkers.py      # Native chunking
│   ├── privacy_detector.py  # LLM privacy classification
│   ├── metadata_extractor.py  # Metadata extraction
│   └── pipeline.py      # LangGraph orchestration
├── vectordb/            # ChromaDB operations
├── utils/               # Utility functions
└── main.py              # Entry point
```

## Metadata Schema

Each chunk stores rich metadata:

- `document_title`: Document name
- `section`: Main section name
- `subsection`: Subsection name (if any)
- `has_private_info`: Boolean flag
- `privacy_level`: "public", "mixed", or "private"
- `num_tokens`: Token count
- `num_words`: Word count
- `char_start`, `char_end`: Character positions
- `chunk_index`: Sequential position
- `source_file`: Relative file path
- `page_number`: For PDFs
- `heading_level`: For markdown headings

## Querying the Knowledge Base

```python
from knowledge_base.vectordb.chroma_store import ChromaStore
from knowledge_base.config.settings import Settings

settings = Settings.load_from_yaml()

store = ChromaStore(
    persist_directory=settings.paths.vector_db,
    embeddings_model=settings.embeddings.model,
    openrouter_api_key=settings.openrouter_api_key,
    openrouter_base_url=settings.openrouter_base_url,
)

# The vector_store is a LangChain Chroma instance
results = store.vector_store.similarity_search(
    query="How to make scallops?",
    k=5,
    filter={"has_private_info": False}  # Public only
)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

## Privacy Detection

The system uses an LLM to classify each chunk as:
- **Public**: No private content
- **Mixed**: Contains both public and private content
- **Private**: Entirely private content

Private indicators include:
- Explicit markers ("Restricted Section", "Internal", "Secret")
- Methodology sections in research papers
- Internal financial data
- Staff information

## Development

Run quality checks:
```bash
uv run ruff check --fix .
uv run ruff format .
uv run pyrefly check
```

## Testing
```bash
uv run pytest tests/knowledge_base/
```