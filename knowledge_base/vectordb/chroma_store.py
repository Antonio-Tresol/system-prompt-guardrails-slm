import logging
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from knowledge_base.schemas.chunk_metadata import ChunkMetadata

logger = logging.getLogger(__name__)


class ChromaStore:
    """ChromaDB operations for knowledge base."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "knowledge_base",
        embeddings_model: str = "google/gemini-embedding-001",
        openrouter_api_key: str = "",
        openrouter_base_url: str = "",
    ) -> None:
        """Initialize ChromaDB store.

        Args:
            persist_directory: Path to persist ChromaDB.
            collection_name: Name of the collection.
            embeddings_model: Model to use for embeddings.
            openrouter_api_key: API key for OpenRouter.
            openrouter_base_url: Base URL for OpenRouter.
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.embeddings = OpenAIEmbeddings(
            model=embeddings_model,
            api_key=openrouter_api_key,
            base_url=openrouter_base_url,
        )

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )

        self._client = chromadb.PersistentClient(path=str(self.persist_directory))

    def add_chunks(self, texts: list[str], metadatas: list[ChunkMetadata], ids: list[str]) -> int:
        """Add chunks to the vector store.

        Args:
            texts: List of chunk texts.
            metadatas: List of chunk metadata objects.
            ids: List of unique IDs for each chunk.

        Returns:
            Number of chunks added successfully.
        """
        try:
            metadata_dicts = [meta.to_dict() for meta in metadatas]

            self.vector_store.add_texts(texts=texts, metadatas=metadata_dicts, ids=ids)

            logger.info(f"Successfully added {len(texts)} chunks to {self.collection_name}")
            return len(texts)
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            return 0

    def collection_exists(self) -> bool:
        """Check if the collection exists.

        Returns:
            True if collection exists, False otherwise.
        """
        try:
            collections = self._client.list_collections()
            return any(c.name == self.collection_name for c in collections)
        except Exception:
            return False

    def get_collection_stats(self) -> dict[str, int | str]:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection statistics.
        """
        try:
            collection = self._client.get_collection(name=self.collection_name)
            count = collection.count()

            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"total_chunks": 0, "collection_name": self.collection_name}
