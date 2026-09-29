from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from backend.app.config import settings
from backend.app.logging_config import logger
from backend.app.schemas import SourceChunk


class VectorStoreManager:
    """Manages persistent ChromaDB vector store operations."""

    def __init__(self):
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

    @property
    def client(self) -> chromadb.PersistentClient:
        """Initializes and returns the ChromaDB persistent client."""
        if self._client is None:
            logger.info(f"Initializing persistent ChromaDB client at: {settings.CHROMA_DB_PATH}")
            self._client = chromadb.PersistentClient(
                path=str(settings.CHROMA_DB_PATH),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        return self._client

    @property
    def collection(self):
        """Gets or creates the ChromaDB collection for document chunks."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def add_chunks(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """Add document chunks and their embeddings to the vector collection."""
        if not ids:
            return
        
        logger.info(f"Upserting {len(ids)} chunks into ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}'")
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def search_similar(self, query_embedding: List[float], top_k: int = 3) -> List[SourceChunk]:
        """Search for top_k most similar document chunks.

        Args:
            query_embedding: Vector embedding of the search query.
            top_k: Number of top relevant results to retrieve.

        Returns:
            List of SourceChunk schemas.
        """
        count = self.count()
        if count == 0:
            logger.warning("Vector store is empty. No documents to retrieve.")
            return []

        # Ensure top_k does not exceed total items in collection
        n_results = min(top_k, count)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        source_chunks: List[SourceChunk] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return source_chunks

        retrieved_ids = results["ids"][0]
        retrieved_docs = results["documents"][0] if results.get("documents") else []
        retrieved_metadatas = results["metadatas"][0] if results.get("metadatas") else []
        retrieved_distances = results["distances"][0] if results.get("distances") else []

        for i, chunk_id in enumerate(retrieved_ids):
            content = retrieved_docs[i] if i < len(retrieved_docs) else ""
            meta = retrieved_metadatas[i] if i < len(retrieved_metadatas) else {}
            dist = retrieved_distances[i] if i < len(retrieved_distances) else None
            
            # Cosine distance ranges from 0 (identical) to 2 (opposite)
            # Relevance score approx 1 - distance (clamped between 0 and 1)
            relevance = None
            if dist is not None:
                relevance = max(0.0, min(1.0, round(1.0 - float(dist), 4)))

            source_chunks.append(
                SourceChunk(
                    source=str(meta.get("source", "unknown")),
                    chunk_id=chunk_id,
                    content=content,
                    distance=round(float(dist), 4) if dist is not None else None,
                    relevance_score=relevance
                )
            )

        return source_chunks

    def count(self) -> int:
        """Return the count of indexed chunks in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error reading collection count: {e}")
            return 0

    def reset_collection(self) -> None:
        """Deletes and recreates the collection for fresh ingestion."""
        logger.warning(f"Resetting collection '{settings.CHROMA_COLLECTION_NAME}'")
        try:
            self.client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
        except Exception:
            pass
        self._collection = None
        _ = self.collection


vector_store = VectorStoreManager()
