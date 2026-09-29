from typing import List, Optional
from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    """Schema representing a retrieved context chunk."""
    source: str = Field(..., description="The source file name or path")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="Text content of the retrieved chunk")
    distance: Optional[float] = Field(None, description="Similarity distance score (lower means more similar)")
    relevance_score: Optional[float] = Field(None, description="Calculated similarity score (0.0 to 1.0)")


class ChatRequest(BaseModel):
    """Request payload for the /api/chat endpoint."""
    query: str = Field(
        ...,
        min_length=1,
        description="The question or prompt to ask the RAG system",
        json_schema_extra={"example": "How do we reduce LLM API costs?"}
    )


class ChatResponse(BaseModel):
    """Response payload for the /api/chat endpoint."""
    query: str = Field(..., description="The user's original query")
    answer: str = Field(..., description="Generated answer based on retrieved documents or cache")
    sources: List[SourceChunk] = Field(
        default_factory=list,
        description="List of top retrieved document chunks used for context"
    )
    cache_hit: bool = Field(
        False,
        description="Indicates whether the answer was served from the semantic cache"
    )
    cache_similarity: float = Field(
        0.0,
        description="Semantic similarity score (0.0 - 1.0) of the matched cached query"
    )
    complexity_score: Optional[int] = Field(
        None,
        description="Deterministic query complexity score (null for cache hits)"
    )
    complexity_level: Optional[str] = Field(
        None,
        description="Complexity classification (simple, medium, complex, not_analyzed)"
    )
    selected_model: Optional[str] = Field(
        None,
        description="The LLM model selected for generating the answer (or not_used for cache hits)"
    )
    routing_reason: Optional[str] = Field(
        None,
        description="Explanation of the routing decision"
    )


class RoutingDecision(BaseModel):
    """Internal schema representing the routing decision for a query."""
    complexity_score: int
    complexity_level: str
    selected_model: str
    routing_reason: str
    factors: List[str] = Field(default_factory=list)


class AnalyticsResponse(BaseModel):
    """Response payload for the /api/analytics endpoint."""
    total_requests: int = Field(..., description="Total chat requests processed")
    successful_requests: int = Field(0, description="Total successful requests processed")
    failed_requests: int = Field(0, description="Total failed requests")
    cache_hits: int = Field(..., description="Total semantic cache hits")
    cache_misses: int = Field(..., description="Total semantic cache misses")
    cache_hit_rate: float = Field(..., description="Semantic cache hit rate (0.0 to 1.0)")
    small_model_requests: int = Field(..., description="Inferences routed to small model")
    large_model_requests: int = Field(..., description="Inferences routed to large model")
    simple_queries: int = Field(..., description="Queries classified as simple")
    medium_queries: int = Field(..., description="Queries classified as medium")
    complex_queries: int = Field(..., description="Queries classified as complex")
    average_latency_ms: float = Field(..., description="Average request latency in milliseconds")
    p50_latency_ms: float = Field(0.0, description="50th percentile (median) request latency in ms")
    p95_latency_ms: float = Field(0.0, description="95th percentile request latency in ms")
    average_cache_latency_ms: float = Field(0.0, description="Average cache lookup latency in ms")
    average_retrieval_latency_ms: float = Field(0.0, description="Average vector retrieval latency in ms")
    average_generation_latency_ms: float = Field(0.0, description="Average LLM generation latency in ms")
    llm_calls: int = Field(0, description="Total LLM generation calls executed")
    llm_calls_avoided_by_cache: int = Field(0, description="Total LLM generation calls saved by cache hits")
    actual_api_cost: float = Field(0.0, description="Actual commercial API cost in USD (strictly 0.0 for local RAG)")
    estimated_compute_cost: Optional[float] = Field(None, description="Estimated compute cost in USD (null unless configured)")


class ModelAnalyticsItem(BaseModel):
    """Schema representing per-model usage and latency statistics."""
    requests: int = Field(..., description="Total requests processed by model")
    average_latency_ms: float = Field(..., description="Average latency in milliseconds")
    errors: int = Field(0, description="Total error requests for model")


class LatencyBreakdownResponse(BaseModel):
    """Schema representing latency breakdown across pipeline stages."""
    average_total_latency_ms: float = Field(..., description="Average total request latency in ms")
    average_cache_lookup_ms: float = Field(..., description="Average cache lookup latency in ms")
    average_routing_ms: float = Field(..., description="Average query routing latency in ms")
    average_retrieval_ms: float = Field(..., description="Average document retrieval latency in ms")
    average_generation_ms: float = Field(..., description="Average LLM generation latency in ms")


class CacheStatsResponse(BaseModel):
    """Response payload for the /api/cache/stats endpoint."""
    total_requests: int = Field(..., description="Total requests processed by the caching layer")
    cache_hits: int = Field(..., description="Number of cache hits")
    cache_misses: int = Field(..., description="Number of cache misses")
    cache_hit_rate: float = Field(..., description="Proportion of requests resolved from cache (0.0 to 1.0)")
    cached_entries: int = Field(..., description="Total number of cached items in vector store")


class CacheDeleteResponse(BaseModel):
    """Response payload for DELETE /api/cache endpoint."""
    status: str
    message: str
    cleared_entries: int


class HealthResponse(BaseModel):
    """Response payload for root / and health endpoints."""
    status: str
    project: str
    version: str
    indexed_chunks: int
    embedding_model: str
    llm_model: str
    ollama_status: str


class IngestResponse(BaseModel):
    """Response payload for document ingestion."""
    status: str
    files_processed: List[str]
    total_chunks: int
    message: str

