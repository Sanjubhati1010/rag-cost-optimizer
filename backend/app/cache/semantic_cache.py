import json
import hashlib
import threading
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.app.config import settings
from backend.app.logging_config import logger
from backend.app.schemas import SourceChunk
from backend.app.embeddings import embedding_service


class CacheStatsTracker:
    """Thread-safe in-memory statistics tracker for semantic cache operations."""

    def __init__(self):
        self._lock = threading.Lock()
        self._total_requests: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def record_hit(self) -> None:
        with self._lock:
            self._total_requests += 1
            self._cache_hits += 1

    def record_miss(self) -> None:
        with self._lock:
            self._total_requests += 1
            self._cache_misses += 1

    def get_stats(self, cached_entries: int) -> Dict[str, Any]:
        with self._lock:
            hit_rate = (
                round(self._cache_hits / self._total_requests, 4)
                if self._total_requests > 0
                else 0.0
            )
            return {
                "total_requests": self._total_requests,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": hit_rate,
                "cached_entries": cached_entries,
            }

    def reset(self) -> None:
        with self._lock:
            self._total_requests = 0
            self._cache_hits = 0
            self._cache_misses = 0


class SemanticCacheManager:
    """Manages semantic vector caching using persistent ChromaDB."""

    def __init__(self):
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self.stats = CacheStatsTracker()

    @property
    def client(self) -> chromadb.PersistentClient:
        """Lazy-loads persistent ChromaDB client for caching."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(settings.CHROMA_DB_PATH),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self):
        """Gets or creates the dedicated semantic_cache collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=settings.SEMANTIC_CACHE_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def count(self) -> int:
        """Returns the number of cached entries in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error reading cache collection count: {e}")
            return 0

    def get_cached_response(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        threshold: Optional[float] = None
    ) -> Optional[Tuple[str, List[SourceChunk], float]]:
        """Searches the semantic cache for a sufficiently similar past query.

        Args:
            query: The user query string.
            query_embedding: Precomputed query embedding vector (optional).
            threshold: Minimum similarity threshold (defaults to settings.SEMANTIC_CACHE_THRESHOLD).

        Returns:
            Tuple of (answer, sources, similarity) on Cache HIT.
            None on Cache MISS or if caching is disabled.
        """
        if not settings.SEMANTIC_CACHE_ENABLED:
            logger.info("Semantic cache is disabled via configuration.")
            return None

        sim_threshold = threshold if threshold is not None else settings.SEMANTIC_CACHE_THRESHOLD

        try:
            total_cached = self.count()
            if total_cached == 0:
                logger.info("[SEMANTIC CACHE] MISS similarity=0.00 (cache is empty)")
                self.stats.record_miss()
                return None

            # Compute embedding if not provided
            if query_embedding is None:
                query_embedding = embedding_service.get_embedding(query.strip())

            # Query top 1 closest cached item
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=["documents", "metadatas", "distances"]
            )

            if not results or not results.get("ids") or not results["ids"][0]:
                logger.info("[SEMANTIC CACHE] MISS similarity=0.00")
                self.stats.record_miss()
                return None

            matched_id = results["ids"][0][0]
            matched_query = results["documents"][0][0] if results.get("documents") else ""
            metadata = results["metadatas"][0][0] if results.get("metadatas") else {}
            distance = results["distances"][0][0] if results.get("distances") else 1.0

            # Cosine distance to similarity: similarity = 1.0 - distance
            similarity = max(0.0, min(1.0, round(1.0 - float(distance), 4)))

            if similarity >= sim_threshold:
                logger.info(
                    f"[SEMANTIC CACHE] HIT similarity={similarity:.2f} "
                    f"(matched cached query='{matched_query}', id='{matched_id}')"
                )
                self.stats.record_hit()

                # Increment hit count in metadata
                current_hits = int(metadata.get("hit_count", 0)) + 1
                try:
                    updated_meta = dict(metadata)
                    updated_meta["hit_count"] = current_hits
                    self.collection.update(
                        ids=[matched_id],
                        metadatas=[updated_meta]
                    )
                except Exception as meta_err:
                    logger.warning(f"Could not update cache hit count: {meta_err}")

                # Deserialize cached sources
                cached_sources: List[SourceChunk] = []
                sources_json = metadata.get("sources_json")
                if sources_json:
                    try:
                        raw_sources = json.loads(sources_json)
                        for s in raw_sources:
                            cached_sources.append(SourceChunk(**s))
                    except Exception as parse_err:
                        logger.warning(f"Could not deserialize cached sources: {parse_err}")

                cached_answer = metadata.get("answer", "")
                return cached_answer, cached_sources, similarity

            else:
                logger.info(
                    f"[SEMANTIC CACHE] MISS similarity={similarity:.2f} "
                    f"(best match='{matched_query}', threshold={sim_threshold:.2f})"
                )
                self.stats.record_miss()
                return None

        except Exception as e:
            logger.error(f"Error during semantic cache retrieval: {e}", exc_info=True)
            # Never block RAG execution on cache failure
            return None

    def store_cached_response(
        self,
        query: str,
        query_embedding: List[float],
        answer: str,
        sources: List[SourceChunk],
        threshold: Optional[float] = None
    ) -> None:
        """Stores a new query, embedding, answer, and metadata into the semantic cache.

        Args:
            query: The user query string.
            query_embedding: Embedding vector of the query.
            answer: Generated answer from the LLM.
            sources: List of SourceChunk objects used for context.
            threshold: Duplicate prevention similarity threshold.
        """
        if not settings.SEMANTIC_CACHE_ENABLED:
            return

        sim_threshold = threshold if threshold is not None else settings.SEMANTIC_CACHE_THRESHOLD

        try:
            trimmed_query = query.strip()
            total_cached = self.count()

            # Check if an identical or near-duplicate entry already exists
            if total_cached > 0:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=1,
                    include=["distances"]
                )
                if results and results.get("distances") and results["distances"][0]:
                    best_distance = results["distances"][0][0]
                    best_sim = max(0.0, min(1.0, round(1.0 - float(best_distance), 4)))
                    if best_sim >= sim_threshold:
                        logger.info(
                            f"[SEMANTIC CACHE] SKIPPED store: similar entry already exists "
                            f"(similarity={best_sim:.2f} >= threshold={sim_threshold:.2f})"
                        )
                        return

            # Generate deterministic cache ID based on query text
            query_hash = hashlib.sha256(trimmed_query.lower().encode("utf-8")).hexdigest()[:12]
            cache_id = f"cache_{query_hash}_{int(datetime.now().timestamp())}"

            # Serialize sources to JSON
            serialized_sources = json.dumps([s.model_dump() for s in sources])

            cache_metadata = {
                "query": trimmed_query,
                "answer": answer,
                "sources_json": serialized_sources,
                "model": settings.OLLAMA_MODEL,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "hit_count": 0,
            }

            self.collection.add(
                ids=[cache_id],
                embeddings=[query_embedding],
                documents=[trimmed_query],
                metadatas=[cache_metadata],
            )

            logger.info(f'[SEMANTIC CACHE] STORED query="{trimmed_query}" (id={cache_id})')

        except Exception as e:
            logger.error(f"Failed to store entry in semantic cache: {e}", exc_info=True)
            # Do not raise; cache store failure must not fail user requests

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on semantic cache usage."""
        cached_count = self.count()
        return self.stats.get_stats(cached_entries=cached_count)

    def clear(self) -> int:
        """Clears all entries from the semantic cache and resets in-memory statistics.

        Returns:
            Number of entries cleared.
        """
        try:
            cleared_count = self.count()
            logger.warning(
                f"Resetting semantic cache collection '{settings.SEMANTIC_CACHE_COLLECTION_NAME}' "
                f"({cleared_count} entries to be cleared)"
            )
            try:
                self.client.delete_collection(name=settings.SEMANTIC_CACHE_COLLECTION_NAME)
            except Exception:
                pass
            self._collection = None
            _ = self.collection
            self.stats.reset()
            logger.info("[SEMANTIC CACHE] Cache cleared successfully.")
            return cleared_count
        except Exception as e:
            logger.error(f"Error clearing semantic cache: {e}", exc_info=True)
            return 0


semantic_cache = SemanticCacheManager()
