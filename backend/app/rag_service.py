import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from backend.app.config import settings
from backend.app.logging_config import logger
from backend.app.schemas import ChatResponse, SourceChunk, RoutingDecision
from backend.app.embeddings import embedding_service
from backend.app.vector_store import vector_store
from backend.app.llm import ollama_client
from backend.app.cache import semantic_cache
from backend.app.router import query_router
from backend.app.analytics import analytics_tracker
from backend.app.monitoring.telemetry import TelemetryRecord


class RAGService:
    """Core RAG service orchestrating semantic caching, query routing, vector retrieval, local LLM synthesis, and latency telemetry."""

    def __init__(self):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_client = ollama_client
        self.cache = semantic_cache
        self.router = query_router
        self.analytics = analytics_tracker

    def answer_query(self, query: str, top_k: int = settings.TOP_K_RETRIEVAL) -> ChatResponse:
        """Processes a user query through the RAG pipeline with Semantic Caching & Cost-Aware Routing.

        Request Flow:
        1. Clean and validate query string.
        2. Compute query embedding vector (Sentence Transformers all-MiniLM-L6-v2).
        3. If Semantic Caching is enabled:
           - Search semantic_cache collection in ChromaDB.
           - If similarity >= threshold (CACHE HIT):
             * Log "[ROUTER] cache hit - routing skipped"
             * Return cached answer and sources directly.
             * DO NOT execute query complexity routing, vector retrieval, or call the LLM.
        4. On CACHE MISS (or cache disabled):
           - Analyze query complexity deterministically and route to appropriate model.
           - Retrieve top-K relevant chunks from persistent ChromaDB.
           - Synthesize answer with selected local Ollama LLM.
           - Store new query + embedding + answer + sources into semantic cache.
           - Record latency and telemetry metrics.
           - Return structured ChatResponse.

        Args:
            query: The user's query string.
            top_k: Number of document chunks to retrieve (defaults to 3).

        Returns:
            ChatResponse schema model.
        """
        req_start = time.perf_counter()
        request_id = str(uuid.uuid4())
        trimmed_query = query.strip()
        logger.info(f"[REQ:{request_id[:8]}] Processing RAG query: '{trimmed_query}' (top_k={top_k})")

        cache_lookup_latency_ms: Optional[float] = None
        routing_latency_ms: Optional[float] = None
        retrieval_latency_ms: Optional[float] = None
        routing_decision: Optional[RoutingDecision] = None

        # Step 1: Compute query embedding
        try:
            logger.info(f"Computing query embedding using '{settings.EMBEDDING_MODEL_NAME}'...")
            query_embedding = self.embedding_service.get_embedding(trimmed_query)
        except Exception as emb_err:
            total_lat = round((time.perf_counter() - req_start) * 1000, 2)
            logger.error(f"[REQ:{request_id[:8]}] Embedding computation failed: {emb_err}", exc_info=True)
            self.analytics.record_telemetry(
                TelemetryRecord(
                    request_id=request_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    query=trimmed_query,
                    status="error",
                    error_stage="embedding",
                    error_message=str(emb_err),
                    total_request_latency_ms=total_lat,
                    actual_api_cost=0.0
                )
            )
            raise

        # Step 2: Semantic Cache Lookup (Phase 2)
        if settings.SEMANTIC_CACHE_ENABLED:
            cache_start = time.perf_counter()
            try:
                cached_result = self.cache.get_cached_response(
                    query=trimmed_query,
                    query_embedding=query_embedding
                )
                cache_lookup_latency_ms = round((time.perf_counter() - cache_start) * 1000, 2)

                if cached_result is not None:
                    cached_answer, cached_sources, similarity = cached_result
                    logger.info(
                        f"[SEMANTIC CACHE] Returning cached answer (similarity={similarity:.2f}) "
                        f"- Skipping RAG retrieval & LLM generation."
                    )
                    logger.info("[ROUTER] cache hit - routing skipped")
                    
                    total_latency_ms = round((time.perf_counter() - req_start) * 1000, 2)
                    self.analytics.record_telemetry(
                        TelemetryRecord(
                            request_id=request_id,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            query=trimmed_query,
                            status="success",
                            cache_hit=True,
                            cache_similarity=similarity,
                            cache_lookup_latency_ms=cache_lookup_latency_ms,
                            complexity_score=None,
                            complexity_level="not_analyzed",
                            selected_model="not_used",
                            routing_reason="Cache hit; routing skipped",
                            routing_latency_ms=None,
                            retrieval_latency_ms=None,
                            number_of_sources=len(cached_sources),
                            retrieval_top_k=top_k,
                            model=None,
                            generation_latency_ms=None,
                            input_tokens=None,
                            output_tokens=None,
                            total_tokens=None,
                            load_duration_ms=None,
                            prompt_eval_duration_ms=None,
                            eval_duration_ms=None,
                            total_request_latency_ms=total_latency_ms,
                            actual_api_cost=0.0,
                            estimated_compute_cost=None,
                            cost_calculation_method="local_ollama_zero_api_cost"
                        )
                    )

                    return ChatResponse(
                        query=trimmed_query,
                        answer=cached_answer,
                        sources=cached_sources,
                        cache_hit=True,
                        cache_similarity=similarity,
                        complexity_score=None,
                        complexity_level="not_analyzed",
                        selected_model="not_used",
                        routing_reason="Cache hit; routing skipped"
                    )
            except Exception as cache_err:
                cache_lookup_latency_ms = round((time.perf_counter() - cache_start) * 1000, 2)
                logger.error(f"Semantic cache lookup failed, falling back to standard RAG: {cache_err}")

        # Step 3: Cost-Aware Intelligent Routing (Phase 3)
        route_start = time.perf_counter()
        try:
            routing_decision = self.router.route_query(trimmed_query)
            routing_latency_ms = round((time.perf_counter() - route_start) * 1000, 2)
        except Exception as route_err:
            routing_latency_ms = round((time.perf_counter() - route_start) * 1000, 2)
            logger.error(f"Routing failure, falling back to default model: {route_err}")
            routing_decision = RoutingDecision(
                complexity_score=0,
                complexity_level="simple",
                selected_model=settings.SMALL_MODEL,
                routing_reason="Fallback to small model due to router error",
                factors=["Router fallback"]
            )

        # Step 4: Check if document repository is indexed
        indexed_count = self.vector_store.count()
        if indexed_count == 0:
            logger.warning("No documents are currently indexed in the vector database.")
            total_latency_ms = round((time.perf_counter() - req_start) * 1000, 2)
            self.analytics.record_telemetry(
                TelemetryRecord(
                    request_id=request_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    query=trimmed_query,
                    status="success",
                    cache_hit=False,
                    cache_similarity=0.0,
                    cache_lookup_latency_ms=cache_lookup_latency_ms,
                    complexity_score=routing_decision.complexity_score,
                    complexity_level=routing_decision.complexity_level,
                    selected_model=routing_decision.selected_model,
                    routing_reason=routing_decision.routing_reason,
                    routing_latency_ms=routing_latency_ms,
                    retrieval_latency_ms=0.0,
                    number_of_sources=0,
                    retrieval_top_k=top_k,
                    model="not_used",
                    generation_latency_ms=0.0,
                    total_request_latency_ms=total_latency_ms,
                    actual_api_cost=0.0,
                    estimated_compute_cost=None,
                    cost_calculation_method="local_ollama_zero_api_cost"
                )
            )
            return ChatResponse(
                query=trimmed_query,
                answer="I am sorry, but no documents have been indexed yet in the database. Please add .txt files to backend/data/documents/ and ingest them.",
                sources=[],
                cache_hit=False,
                cache_similarity=0.0,
                complexity_score=routing_decision.complexity_score,
                complexity_level=routing_decision.complexity_level,
                selected_model=routing_decision.selected_model,
                routing_reason=routing_decision.routing_reason
            )

        # Step 5: Retrieve top K document chunks from ChromaDB
        retrieval_start = time.perf_counter()
        try:
            logger.info(f"Retrieving top {top_k} most relevant chunks from ChromaDB...")
            sources: List[SourceChunk] = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k
            )
            retrieval_latency_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)
            logger.info(f"Retrieved {len(sources)} source chunk(s) in {retrieval_latency_ms}ms.")
        except Exception as ret_err:
            retrieval_latency_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)
            total_lat = round((time.perf_counter() - req_start) * 1000, 2)
            logger.error(f"[REQ:{request_id[:8]}] Vector retrieval failed: {ret_err}", exc_info=True)
            self.analytics.record_telemetry(
                TelemetryRecord(
                    request_id=request_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    query=trimmed_query,
                    status="error",
                    error_stage="retrieval",
                    error_message=str(ret_err),
                    cache_hit=False,
                    cache_lookup_latency_ms=cache_lookup_latency_ms,
                    complexity_score=routing_decision.complexity_score,
                    complexity_level=routing_decision.complexity_level,
                    selected_model=routing_decision.selected_model,
                    routing_latency_ms=routing_latency_ms,
                    retrieval_latency_ms=retrieval_latency_ms,
                    retrieval_top_k=top_k,
                    total_request_latency_ms=total_lat,
                    actual_api_cost=0.0
                )
            )
            raise

        # Step 6: Synthesize answer with routed Ollama LLM
        try:
            answer, llm_meta = self.llm_client.generate_answer_with_telemetry(
                query=trimmed_query,
                context_chunks=sources,
                model=routing_decision.selected_model
            )
        except Exception as llm_err:
            total_lat = round((time.perf_counter() - req_start) * 1000, 2)
            logger.error(f"[REQ:{request_id[:8]}] LLM generation failed: {llm_err}", exc_info=True)
            self.analytics.record_telemetry(
                TelemetryRecord(
                    request_id=request_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    query=trimmed_query,
                    status="error",
                    error_stage="llm_generation",
                    error_message=str(llm_err),
                    cache_hit=False,
                    cache_lookup_latency_ms=cache_lookup_latency_ms,
                    complexity_score=routing_decision.complexity_score,
                    complexity_level=routing_decision.complexity_level,
                    selected_model=routing_decision.selected_model,
                    routing_reason=routing_decision.routing_reason,
                    routing_latency_ms=routing_latency_ms,
                    retrieval_latency_ms=retrieval_latency_ms,
                    number_of_sources=len(sources),
                    retrieval_top_k=top_k,
                    model=routing_decision.selected_model,
                    total_request_latency_ms=total_lat,
                    actual_api_cost=0.0
                )
            )
            raise

        # Step 7: Store in Semantic Cache on Cache MISS
        if settings.SEMANTIC_CACHE_ENABLED:
            try:
                self.cache.store_cached_response(
                    query=trimmed_query,
                    query_embedding=query_embedding,
                    answer=answer,
                    sources=sources
                )
            except Exception as store_err:
                logger.error(f"Failed to store response in semantic cache: {store_err}")

        total_latency_ms = round((time.perf_counter() - req_start) * 1000, 2)

        # Record full telemetry
        self.analytics.record_telemetry(
            TelemetryRecord(
                request_id=request_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                query=trimmed_query,
                status="success",
                cache_hit=False,
                cache_similarity=0.0,
                cache_lookup_latency_ms=cache_lookup_latency_ms,
                complexity_score=routing_decision.complexity_score,
                complexity_level=routing_decision.complexity_level,
                selected_model=routing_decision.selected_model,
                routing_reason=routing_decision.routing_reason,
                routing_latency_ms=routing_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                number_of_sources=len(sources),
                retrieval_top_k=top_k,
                model=llm_meta.get("model", routing_decision.selected_model),
                generation_latency_ms=llm_meta.get("generation_latency_ms"),
                input_tokens=llm_meta.get("input_tokens"),
                output_tokens=llm_meta.get("output_tokens"),
                total_tokens=llm_meta.get("total_tokens"),
                load_duration_ms=llm_meta.get("load_duration_ms"),
                prompt_eval_duration_ms=llm_meta.get("prompt_eval_duration_ms"),
                eval_duration_ms=llm_meta.get("eval_duration_ms"),
                total_request_latency_ms=total_latency_ms,
                actual_api_cost=0.0,
                estimated_compute_cost=None,
                cost_calculation_method="local_ollama_zero_api_cost"
            )
        )

        return ChatResponse(
            query=trimmed_query,
            answer=answer,
            sources=sources,
            cache_hit=False,
            cache_similarity=0.0,
            complexity_score=routing_decision.complexity_score,
            complexity_level=routing_decision.complexity_level,
            selected_model=routing_decision.selected_model,
            routing_reason=routing_decision.routing_reason
        )


rag_service = RAGService()

