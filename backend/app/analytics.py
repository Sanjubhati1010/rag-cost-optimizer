import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from backend.app.monitoring.telemetry import TelemetryRecord
from backend.app.monitoring.storage import telemetry_storage
from backend.app.monitoring.metrics import metrics_service


class AnalyticsTracker:
    """Production analytics tracker backed by persistent SQLite TelemetryStorage and MetricsService."""

    def __init__(self, storage=telemetry_storage, metrics=metrics_service):
        self.storage = storage
        self.metrics = metrics

    def record_telemetry(self, record: TelemetryRecord) -> None:
        """Saves a comprehensive request telemetry record."""
        self.storage.save_record(record)

    def record_request(
        self,
        cache_hit: bool,
        complexity_level: Optional[str] = None,
        selected_model: Optional[str] = None,
        latency_ms: float = 0.0,
        query: str = "unknown",
        request_id: Optional[str] = None,
        status: str = "success",
        error_stage: Optional[str] = None,
        error_message: Optional[str] = None,
        complexity_score: Optional[int] = None,
        routing_reason: Optional[str] = None,
        routing_latency_ms: Optional[float] = None,
        retrieval_latency_ms: Optional[float] = None,
        generation_latency_ms: Optional[float] = None,
        cache_lookup_latency_ms: Optional[float] = None,
        cache_similarity: float = 0.0,
        number_of_sources: int = 0,
        retrieval_top_k: int = 3,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        load_duration_ms: Optional[float] = None,
        prompt_eval_duration_ms: Optional[float] = None,
        eval_duration_ms: Optional[float] = None,
    ) -> TelemetryRecord:
        """Backward-compatible helper to record telemetry for a single processed request."""
        rec_id = request_id or str(uuid.uuid4())
        record = TelemetryRecord(
            request_id=rec_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
            status=status,
            error_stage=error_stage,
            error_message=error_message,
            cache_hit=cache_hit,
            cache_similarity=cache_similarity,
            cache_lookup_latency_ms=cache_lookup_latency_ms,
            complexity_score=complexity_score,
            complexity_level=complexity_level or ("not_analyzed" if cache_hit else "simple"),
            selected_model=selected_model or ("not_used" if cache_hit else None),
            routing_reason=routing_reason or ("Cache hit; routing skipped" if cache_hit else None),
            routing_latency_ms=routing_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            number_of_sources=number_of_sources,
            retrieval_top_k=retrieval_top_k,
            model=selected_model if (selected_model and selected_model != "not_used") else None,
            generation_latency_ms=generation_latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            load_duration_ms=load_duration_ms,
            prompt_eval_duration_ms=prompt_eval_duration_ms,
            eval_duration_ms=eval_duration_ms,
            total_request_latency_ms=round(latency_ms, 2),
            actual_api_cost=0.0,
            estimated_compute_cost=None,
            cost_calculation_method="local_ollama_zero_api_cost"
        )
        self.record_telemetry(record)
        return record

    def get_metrics(self) -> Dict[str, Any]:
        """Returns aggregated analytics metrics calculated across persistent request records."""
        return self.metrics.get_summary_metrics()

    def get_model_analytics(self) -> Dict[str, Dict[str, Any]]:
        """Returns per-model breakdown stats."""
        return self.metrics.get_model_analytics()

    def get_latency_breakdown(self) -> Dict[str, float]:
        """Returns latency breakdown stats."""
        return self.metrics.get_latency_breakdown()

    def get_recent_requests(self, limit: int = 20) -> List[TelemetryRecord]:
        """Returns recent telemetry records."""
        return self.storage.get_recent(limit=limit)

    def reset(self) -> None:
        """Clears persistent telemetry history."""
        self.storage.clear()


analytics_tracker = AnalyticsTracker()

