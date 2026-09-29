from pathlib import Path
from typing import List, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse

from backend.app.config import settings
from backend.app.logging_config import logger
from backend.app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    CacheStatsResponse,
    CacheDeleteResponse,
    AnalyticsResponse,
    ModelAnalyticsItem,
    LatencyBreakdownResponse,
)
from backend.app.monitoring.telemetry import TelemetryRecord
from backend.app.rag_service import rag_service
from backend.app.vector_store import vector_store
from backend.app.ingestion import ingest_documents
from backend.app.llm import ollama_client
from backend.app.cache import semantic_cache
from backend.app.analytics import analytics_tracker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to initialize resources on startup."""
    logger.info("=" * 60)
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.PROJECT_VERSION}")
    logger.info("=" * 60)

    # 1. Check ChromaDB collection status & auto-ingest if empty
    doc_count = vector_store.count()
    logger.info(f"ChromaDB persistent collection status: {doc_count} chunk(s) indexed.")
    if doc_count == 0:
        logger.info("Vector database is empty. Running initial document ingestion from backend/data/documents/...")
        ingest_res = ingest_documents(force_reload=False)
        logger.info(f"Startup ingestion result: {ingest_res.get('message')}")

    # 2. Check Semantic Cache status
    cache_count = semantic_cache.count()
    logger.info(
        f"Semantic Cache status: Enabled={settings.SEMANTIC_CACHE_ENABLED}, "
        f"Threshold={settings.SEMANTIC_CACHE_THRESHOLD}, Cached entries={cache_count}"
    )

    # 3. Check Ollama LLM connectivity
    ollama_health = ollama_client.check_health()
    if ollama_health.get("connected"):
        logger.info(f"Ollama connected successfully. Available models: {ollama_health.get('available_models')}")
        if not ollama_health.get("model_available"):
            logger.warning(
                f"Model '{settings.OLLAMA_MODEL}' not found in Ollama library. "
                f"Please ensure it is downloaded via 'ollama pull {settings.OLLAMA_MODEL}'."
            )
    else:
        logger.warning(
            f"Could not connect to Ollama at {settings.OLLAMA_BASE_URL}. "
            "Please start Ollama service using 'ollama serve' or the Ollama desktop app."
        )

    yield

    logger.info("Shutting down RAG Cost Optimizer application.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Completely local, zero-API-cost RAG system with Semantic Caching and Production Monitoring powered by Sentence Transformers, ChromaDB, and Ollama.",
    lifespan=lifespan
)

# Enable CORS for frontend clients or API consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="Root Health and System Info", response_model=HealthResponse)
def root():
    """Returns basic system info, status, and health metrics."""
    ollama_health = ollama_client.check_health()
    ollama_status = "Connected" if ollama_health.get("connected") else "Disconnected"
    if ollama_health.get("connected") and not ollama_health.get("model_available"):
        ollama_status = f"Connected (Model '{settings.OLLAMA_MODEL}' not downloaded)"

    return HealthResponse(
        status="online",
        project=settings.PROJECT_NAME,
        version=settings.PROJECT_VERSION,
        indexed_chunks=vector_store.count(),
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        llm_model=settings.OLLAMA_MODEL,
        ollama_status=ollama_status
    )


@app.get("/dashboard", summary="Monitoring & Cost Analytics Dashboard", response_class=HTMLResponse)
def get_dashboard():
    """Serves the rich web-based production monitoring dashboard."""
    dashboard_path = Path(__file__).resolve().parent / "app" / "static" / "dashboard.html"
    if dashboard_path.exists():
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard HTML template not found</h1>", status_code=404)


@app.post(
    "/api/chat",
    summary="Chat with RAG System (with Semantic Caching)",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK
)
def chat_endpoint(request: ChatRequest):
    """Processes a user question with Semantic Caching:
    1. Checks semantic_cache for semantically similar questions (>= threshold).
    2. On CACHE HIT: Returns cached answer immediately without invoking RAG retrieval or LLM.
    3. On CACHE MISS: Retrieves top 3 document chunks from ChromaDB, generates answer using
       local Ollama, stores result in cache, and returns answer.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty."
        )

    try:
        response = rag_service.answer_query(request.query)
        return response
    except ConnectionError as e:
        logger.error(f"Connection error during chat processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except TimeoutError as e:
        logger.error(f"Timeout error during LLM generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The local LLM request timed out. Please try again."
        )
    except Exception as e:
        logger.error(f"Internal error processing chat query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating answer: {str(e)}"
        )


@app.get(
    "/api/cache/stats",
    summary="Get Semantic Cache Statistics",
    response_model=CacheStatsResponse,
    status_code=status.HTTP_200_OK
)
def get_cache_stats():
    """Returns real-time statistics of the semantic caching layer, including total requests,
    hits, misses, hit rate, and count of cached entries.
    """
    try:
        stats = semantic_cache.get_stats()
        return CacheStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to retrieve cache stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading cache stats: {str(e)}"
        )


@app.delete(
    "/api/cache",
    summary="Clear Semantic Cache",
    response_model=CacheDeleteResponse,
    status_code=status.HTTP_200_OK
)
def clear_cache():
    """Clears all entries stored in the semantic cache collection and resets in-memory stats."""
    try:
        cleared_count = semantic_cache.clear()
        analytics_tracker.reset()
        return CacheDeleteResponse(
            status="success",
            message="Semantic cache and analytics cleared successfully.",
            cleared_entries=cleared_count
        )
    except Exception as e:
        logger.error(f"Failed to clear semantic cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing cache: {str(e)}"
        )


@app.get(
    "/api/analytics",
    summary="Get Performance, Routing, and Cost Analytics",
    response_model=AnalyticsResponse,
    status_code=status.HTTP_200_OK
)
def get_analytics():
    """Returns real-time aggregated metrics for request latency, cache hit rates,
    model routing distribution, query complexity breakdown, percentiles (p50/p95), and zero commercial API cost confirmation.
    """
    try:
        metrics = analytics_tracker.get_metrics()
        return AnalyticsResponse(**metrics)
    except Exception as e:
        logger.error(f"Failed to retrieve analytics metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading analytics: {str(e)}"
        )


@app.get(
    "/api/analytics/models",
    summary="Get Per-Model Usage and Latency Analytics",
    response_model=Dict[str, ModelAnalyticsItem],
    status_code=status.HTTP_200_OK
)
def get_model_analytics():
    """Returns granular request count, average generation latency, and error rate for each utilized LLM."""
    try:
        models_data = analytics_tracker.get_model_analytics()
        return {m: ModelAnalyticsItem(**d) for m, d in models_data.items()}
    except Exception as e:
        logger.error(f"Failed to retrieve model analytics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading model analytics: {str(e)}"
        )


@app.get(
    "/api/analytics/latency",
    summary="Get Pipeline Latency Breakdown",
    response_model=LatencyBreakdownResponse,
    status_code=status.HTTP_200_OK
)
def get_latency_breakdown():
    """Returns latency averages broken down across Cache Lookup, Routing, Vector Retrieval, and LLM Generation stages."""
    try:
        breakdown = analytics_tracker.get_latency_breakdown()
        return LatencyBreakdownResponse(**breakdown)
    except Exception as e:
        logger.error(f"Failed to retrieve latency breakdown: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading latency breakdown: {str(e)}"
        )


@app.get(
    "/api/analytics/recent",
    summary="Get Recent Request Telemetry Records",
    response_model=List[TelemetryRecord],
    status_code=status.HTTP_200_OK
)
def get_recent_telemetry(
    limit: int = Query(default=20, ge=1, le=100, description="Number of recent telemetry records to retrieve (1-100)")
):
    """Returns the most recent detailed telemetry records with stage timings, token metadata, and status."""
    try:
        records = analytics_tracker.get_recent_requests(limit=limit)
        return records
    except Exception as e:
        logger.error(f"Failed to retrieve recent telemetry: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading recent telemetry records: {str(e)}"
        )


@app.post(
    "/api/ingest",
    summary="Trigger Document Ingestion",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK
)
def trigger_ingest(force_reload: bool = False):
    """Ingests all .txt files from backend/data/documents/ into ChromaDB.
    Set force_reload=true to clear existing index and rebuild from scratch.
    """
    try:
        result = ingest_documents(force_reload=force_reload)
        return IngestResponse(
            status=result["status"],
            files_processed=result["files_processed"],
            total_chunks=result["total_chunks"],
            message=result["message"]
        )
    except Exception as e:
        logger.error(f"Error during document ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}"
        )


@app.get("/health", summary="Health Check")
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "indexed_chunks": vector_store.count(),
        "cached_entries": semantic_cache.count()
    }

