import statistics
from typing import Dict, Any, List, Optional
from backend.app.monitoring.telemetry import TelemetryRecord
from backend.app.monitoring.storage import telemetry_storage


def calculate_percentile(values: List[float], p: float) -> float:
    """Calculates the p-th percentile from a list of numerical values using linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 2)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return round(sorted_vals[f], 2)
    val = sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)
    return round(val, 2)


class MetricsService:
    """Computes aggregated production telemetry, cost, routing, and latency metrics."""

    def __init__(self, storage=telemetry_storage):
        self.storage = storage

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculates global analytics summary across all recorded requests."""
        records = self.storage.get_all_records()
        total_requests = len(records)

        if total_requests == 0:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_hit_rate": 0.0,
                "small_model_requests": 0,
                "large_model_requests": 0,
                "simple_queries": 0,
                "medium_queries": 0,
                "complex_queries": 0,
                "average_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "average_cache_latency_ms": 0.0,
                "average_retrieval_latency_ms": 0.0,
                "average_generation_latency_ms": 0.0,
                "llm_calls": 0,
                "llm_calls_avoided_by_cache": 0,
                "actual_api_cost": 0.0,
                "estimated_compute_cost": None,
            }

        successful = 0
        failed = 0
        cache_hits = 0
        cache_misses = 0
        small_model = 0
        large_model = 0
        simple_q = 0
        medium_q = 0
        complex_q = 0
        llm_calls = 0

        total_latencies: List[float] = []
        cache_latencies: List[float] = []
        retrieval_latencies: List[float] = []
        generation_latencies: List[float] = []

        for r in records:
            if r.status == "success":
                successful += 1
            else:
                failed += 1

            total_latencies.append(r.total_request_latency_ms)

            if r.cache_lookup_latency_ms is not None:
                cache_latencies.append(r.cache_lookup_latency_ms)

            if r.cache_hit:
                cache_hits += 1
            else:
                cache_misses += 1

                # Complexity categorization
                if r.complexity_level == "simple":
                    simple_q += 1
                elif r.complexity_level == "medium":
                    medium_q += 1
                elif r.complexity_level == "complex":
                    complex_q += 1

                # Model routing categorization
                model_str = (r.selected_model or r.model or "").lower()
                if model_str and model_str != "not_used":
                    if "3b" in model_str or "small" in model_str:
                        small_model += 1
                    else:
                        large_model += 1

                # Retrieval latency
                if r.retrieval_latency_ms is not None:
                    retrieval_latencies.append(r.retrieval_latency_ms)

                # Generation latency & LLM calls
                if r.generation_latency_ms is not None:
                    generation_latencies.append(r.generation_latency_ms)
                    llm_calls += 1
                elif r.model and r.model != "not_used" and r.status == "success":
                    llm_calls += 1

        avg_latency = round(sum(total_latencies) / len(total_latencies), 2) if total_latencies else 0.0
        p50_latency = calculate_percentile(total_latencies, 50.0)
        p95_latency = calculate_percentile(total_latencies, 95.0)

        avg_cache_lat = round(sum(cache_latencies) / len(cache_latencies), 2) if cache_latencies else 0.0
        avg_retrieval_lat = round(sum(retrieval_latencies) / len(retrieval_latencies), 2) if retrieval_latencies else 0.0
        avg_gen_lat = round(sum(generation_latencies) / len(generation_latencies), 2) if generation_latencies else 0.0

        hit_rate = round(cache_hits / total_requests, 4)

        return {
            "total_requests": total_requests,
            "successful_requests": successful,
            "failed_requests": failed,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": hit_rate,
            "small_model_requests": small_model,
            "large_model_requests": large_model,
            "simple_queries": simple_q,
            "medium_queries": medium_q,
            "complex_queries": complex_q,
            "average_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "average_cache_latency_ms": avg_cache_lat,
            "average_retrieval_latency_ms": avg_retrieval_lat,
            "average_generation_latency_ms": avg_gen_lat,
            "llm_calls": llm_calls,
            "llm_calls_avoided_by_cache": cache_hits,
            "actual_api_cost": 0.0,
            "estimated_compute_cost": None,
        }

    def get_model_analytics(self) -> Dict[str, Dict[str, Any]]:
        """Calculates per-model performance and latency statistics."""
        records = self.storage.get_all_records()
        model_data: Dict[str, Dict[str, Any]] = {}

        for r in records:
            if r.cache_hit:
                continue

            # Model selected for generation or executed
            target_model = r.selected_model if (r.selected_model and r.selected_model != "not_used") else (
                r.model if (r.model and r.model != "not_used") else None
            )

            if not target_model or target_model == "not_used":
                continue

            if target_model not in model_data:
                model_data[target_model] = {
                    "requests": 0,
                    "latencies": [],
                    "errors": 0
                }

            model_data[target_model]["requests"] += 1
            if r.status == "error":
                model_data[target_model]["errors"] += 1

            if r.generation_latency_ms is not None and r.generation_latency_ms > 0:
                model_data[target_model]["latencies"].append(r.generation_latency_ms)
            elif r.total_request_latency_ms > 0:
                model_data[target_model]["latencies"].append(r.total_request_latency_ms)

        result: Dict[str, Dict[str, Any]] = {}
        for m_name, d in model_data.items():
            avg_lat = round(sum(d["latencies"]) / len(d["latencies"]), 2) if d["latencies"] else 0.0
            result[m_name] = {
                "requests": d["requests"],
                "average_latency_ms": avg_lat,
                "errors": d["errors"]
            }

        return result


    def get_latency_breakdown(self) -> Dict[str, float]:
        """Calculates average latency for each pipeline stage."""
        records = self.storage.get_all_records()
        if not records:
            return {
                "average_total_latency_ms": 0.0,
                "average_cache_lookup_ms": 0.0,
                "average_routing_ms": 0.0,
                "average_retrieval_ms": 0.0,
                "average_generation_ms": 0.0,
            }

        total_lats: List[float] = [r.total_request_latency_ms for r in records if r.total_request_latency_ms is not None]
        cache_lats: List[float] = [r.cache_lookup_latency_ms for r in records if r.cache_lookup_latency_ms is not None]
        routing_lats: List[float] = [r.routing_latency_ms for r in records if r.routing_latency_ms is not None]
        retrieval_lats: List[float] = [r.retrieval_latency_ms for r in records if r.retrieval_latency_ms is not None]
        gen_lats: List[float] = [r.generation_latency_ms for r in records if r.generation_latency_ms is not None]

        return {
            "average_total_latency_ms": round(sum(total_lats) / len(total_lats), 2) if total_lats else 0.0,
            "average_cache_lookup_ms": round(sum(cache_lats) / len(cache_lats), 2) if cache_lats else 0.0,
            "average_routing_ms": round(sum(routing_lats) / len(routing_lats), 2) if routing_lats else 0.0,
            "average_retrieval_ms": round(sum(retrieval_lats) / len(retrieval_lats), 2) if retrieval_lats else 0.0,
            "average_generation_ms": round(sum(gen_lats) / len(gen_lats), 2) if gen_lats else 0.0,
        }


metrics_service = MetricsService()
