from typing import Optional
from pydantic import BaseModel, Field


class TelemetryRecord(BaseModel):
    """Schema representing a single RAG request telemetry and latency record."""
    request_id: str = Field(..., description="Unique UUID for the request")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of the request")
    query: str = Field(..., description="User query text")
    status: str = Field("success", description="Status of the request: success or error")
    error_stage: Optional[str] = Field(None, description="Pipeline stage where failure occurred if error")
    error_message: Optional[str] = Field(None, description="Sanitized error description")

    # Semantic Cache
    cache_hit: bool = Field(False, description="Whether query was served from semantic cache")
    cache_similarity: float = Field(0.0, description="Cosine similarity score for cache lookup")
    cache_lookup_latency_ms: Optional[float] = Field(None, description="Latency of cache check in ms")

    # Routing
    complexity_score: Optional[int] = Field(None, description="Deterministic query complexity score")
    complexity_level: Optional[str] = Field(None, description="Complexity level (simple, medium, complex, not_analyzed)")
    selected_model: Optional[str] = Field(None, description="Model selected for generation or 'not_used'")
    routing_reason: Optional[str] = Field(None, description="Reasoning behind routing decision")
    routing_latency_ms: Optional[float] = Field(None, description="Latency of query complexity analysis in ms")

    # RAG Retrieval
    retrieval_latency_ms: Optional[float] = Field(None, description="Latency of vector retrieval in ms")
    number_of_sources: int = Field(0, description="Count of source chunks retrieved")
    retrieval_top_k: int = Field(3, description="Requested top-K retrieval depth")

    # LLM Generation
    model: Optional[str] = Field(None, description="Actual LLM model invoked or 'not_used'")
    generation_latency_ms: Optional[float] = Field(None, description="LLM inference time in ms")
    input_tokens: Optional[int] = Field(None, description="Prompt token count from Ollama metadata")
    output_tokens: Optional[int] = Field(None, description="Completion token count from Ollama metadata")
    total_tokens: Optional[int] = Field(None, description="Total tokens processed or null")
    load_duration_ms: Optional[float] = Field(None, description="Ollama model load duration in ms")
    prompt_eval_duration_ms: Optional[float] = Field(None, description="Ollama prompt evaluation duration in ms")
    eval_duration_ms: Optional[float] = Field(None, description="Ollama token generation duration in ms")

    # Overall Metrics & Cost
    total_request_latency_ms: float = Field(..., description="End-to-end request duration in ms")
    actual_api_cost: float = Field(0.0, description="Actual commercial API cost in USD (0.0 for local RAG)")
    estimated_compute_cost: Optional[float] = Field(None, description="Estimated compute cost in USD (null unless configured)")
    cost_calculation_method: str = Field("local_ollama_zero_api_cost", description="Method used to calculate cost")
