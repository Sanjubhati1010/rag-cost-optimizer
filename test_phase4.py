import requests
import json
import time

BASE = "http://127.0.0.1:8000"

print("=" * 60)
print("PHASE 4 TELEMETRY, COST ANALYTICS & MONITORING TEST SUITE")
print("=" * 60)

# Step 0: Clear Cache and Analytics
print("\n[STEP 0] Resetting Cache & Analytics...")
r_del = requests.delete(f"{BASE}/api/cache")
print(f"Cache Reset Status: {r_del.status_code}, Response: {r_del.json()}")

# Step 1: Simple query (Cache MISS -> llama3.2:3b)
print("\n[TEST 1] Simple Query: 'What is DBMS?'")
t1_start = time.perf_counter()
r1 = requests.post(f"{BASE}/api/chat", json={"query": "What is DBMS?"})
t1_dur = (time.perf_counter() - t1_start) * 1000
print(f"T1 Status: {r1.status_code} (took {t1_dur:.1f}ms)")
d1 = r1.json()
print("T1 Response:", json.dumps(d1, indent=2))

assert r1.status_code == 200, f"Expected 200, got {r1.status_code}"
assert d1["cache_hit"] is False, "Expected cache_hit to be False"
assert d1["complexity_level"] == "simple", f"Expected simple complexity, got {d1['complexity_level']}"
assert "3b" in d1["selected_model"], f"Expected 3B model, got {d1['selected_model']}"

# Step 2: Complex query (Cache MISS -> mistral:7b)
print("\n[TEST 2] Complex Query: Normalization comparisons")
q_complex = "Compare 1NF, 2NF, 3NF and BCNF with examples, explain functional dependencies and discuss when each normalization level should be used."
t2_start = time.perf_counter()
r2 = requests.post(f"{BASE}/api/chat", json={"query": q_complex})
t2_dur = (time.perf_counter() - t2_start) * 1000
print(f"T2 Status: {r2.status_code} (took {t2_dur:.1f}ms)")
d2 = r2.json()
print("T2 Response Summary:", {
    "cache_hit": d2.get("cache_hit"),
    "complexity_level": d2.get("complexity_level"),
    "complexity_score": d2.get("complexity_score"),
    "selected_model": d2.get("selected_model"),
    "routing_reason": d2.get("routing_reason")
})

assert r2.status_code == 200, f"Expected 200, got {r2.status_code}"
assert d2["cache_hit"] is False, "Expected cache_hit to be False"
assert d2["complexity_level"] == "complex", f"Expected complex, got {d2['complexity_level']}"
assert d2["complexity_score"] >= 7, f"Expected score >= 7, got {d2['complexity_score']}"
assert "mistral:7b" in d2["selected_model"], f"Expected mistral:7b, got {d2['selected_model']}"

# Step 3: Repeat Simple Query (Expect Cache HIT, router skipped)
print("\n[TEST 3] Cache HIT: Repeating 'What is DBMS?'")
t3_start = time.perf_counter()
r3 = requests.post(f"{BASE}/api/chat", json={"query": "What is DBMS?"})
t3_dur = (time.perf_counter() - t3_start) * 1000
print(f"T3 Status: {r3.status_code} (took {t3_dur:.1f}ms)")
d3 = r3.json()
print("T3 Response:", json.dumps(d3, indent=2))

assert r3.status_code == 200
assert d3["cache_hit"] is True, "Expected cache_hit to be True"
assert d3["cache_similarity"] >= 0.90, f"Expected high similarity >= 0.90, got {d3['cache_similarity']}"
assert d3["complexity_score"] is None, "Expected complexity_score to be None on cache hit"
assert d3["complexity_level"] == "not_analyzed", f"Expected not_analyzed, got {d3['complexity_level']}"
assert d3["selected_model"] == "not_used", f"Expected not_used, got {d3['selected_model']}"
assert d3["routing_reason"] == "Cache hit; routing skipped"

# Step 4: GET /api/analytics
print("\n[TEST 4] GET /api/analytics")
r_analytics = requests.get(f"{BASE}/api/analytics")
print("Analytics Status:", r_analytics.status_code)
da = r_analytics.json()
print("Analytics Data:", json.dumps(da, indent=2))

assert r_analytics.status_code == 200
assert da["total_requests"] == 3, f"Expected 3 total requests, got {da['total_requests']}"
assert da["cache_hits"] == 1, f"Expected 1 cache hit, got {da['cache_hits']}"
assert da["cache_misses"] == 2, f"Expected 2 cache misses, got {da['cache_misses']}"
assert round(da["cache_hit_rate"], 2) == 0.33, f"Expected 0.33 hit rate, got {da['cache_hit_rate']}"
assert da["small_model_requests"] == 1, f"Expected 1 small model request, got {da['small_model_requests']}"
assert da["large_model_requests"] == 1, f"Expected 1 large model request, got {da['large_model_requests']}"
assert da["simple_queries"] == 1, f"Expected 1 simple query, got {da['simple_queries']}"
assert da["complex_queries"] == 1, f"Expected 1 complex query, got {da['complex_queries']}"
assert da["llm_calls_avoided_by_cache"] == 1, f"Expected 1 call avoided, got {da['llm_calls_avoided_by_cache']}"
assert da["llm_calls"] == 2, f"Expected 2 LLM calls, got {da['llm_calls']}"
assert da["actual_api_cost"] == 0.0, f"Expected 0.0 actual cost, got {da['actual_api_cost']}"
assert da["estimated_compute_cost"] is None, f"Expected None compute cost, got {da['estimated_compute_cost']}"
assert da["average_latency_ms"] > 0, "Expected positive average latency"
assert da["p50_latency_ms"] > 0, "Expected positive p50 latency"
assert da["p95_latency_ms"] > 0, "Expected positive p95 latency"

# Step 5: GET /api/analytics/models
print("\n[TEST 5] GET /api/analytics/models")
r_models = requests.get(f"{BASE}/api/analytics/models")
print("Models Analytics Status:", r_models.status_code)
dm = r_models.json()
print("Models Analytics Data:", json.dumps(dm, indent=2))

assert r_models.status_code == 200
assert "llama3.2:3b" in dm, "Expected llama3.2:3b in models analytics"
assert "mistral:7b" in dm, "Expected mistral:7b in models analytics"
assert dm["llama3.2:3b"]["requests"] == 1
assert dm["mistral:7b"]["requests"] == 1
assert dm["llama3.2:3b"]["average_latency_ms"] > 0
assert dm["mistral:7b"]["average_latency_ms"] > 0
assert "not_used" not in dm, "not_used should not be listed as a model"

# Step 6: GET /api/analytics/latency
print("\n[TEST 6] GET /api/analytics/latency")
r_latency = requests.get(f"{BASE}/api/analytics/latency")
print("Latency Status:", r_latency.status_code)
dl = r_latency.json()
print("Latency Breakdown Data:", json.dumps(dl, indent=2))

assert r_latency.status_code == 200
assert dl["average_total_latency_ms"] > 0
assert dl["average_cache_lookup_ms"] > 0
assert dl["average_routing_ms"] >= 0
assert dl["average_retrieval_ms"] >= 0
assert dl["average_generation_ms"] > 0

# Step 7: GET /api/analytics/recent?limit=20
print("\n[TEST 7] GET /api/analytics/recent?limit=20")
r_recent = requests.get(f"{BASE}/api/analytics/recent?limit=20")
print("Recent Status:", r_recent.status_code)
d_recent = r_recent.json()
print(f"Recent Records Count: {len(d_recent)}")
assert r_recent.status_code == 200
assert len(d_recent) == 3, f"Expected 3 records, got {len(d_recent)}"

for rec in d_recent:
    print(f"Record [{rec['request_id'][:8]}] | Query: '{rec['query']}' | Cache: {rec['cache_hit']} | Latency: {rec['total_request_latency_ms']}ms | Actual Cost: ${rec['actual_api_cost']}")
    assert rec["request_id"] is not None
    assert rec["timestamp"] is not None
    assert rec["actual_api_cost"] == 0.0
    assert rec["total_request_latency_ms"] > 0
    if rec["input_tokens"] is not None:
        assert isinstance(rec["input_tokens"], int)
    if rec["output_tokens"] is not None:
        assert isinstance(rec["output_tokens"], int)

# Step 8: GET /dashboard
print("\n[TEST 8] GET /dashboard")
r_dash = requests.get(f"{BASE}/dashboard")
print("Dashboard Status:", r_dash.status_code)
assert r_dash.status_code == 200
assert "<!DOCTYPE html>" in r_dash.text
assert "RAG Cost Optimizer" in r_dash.text
assert "stat-total-requests" in r_dash.text
assert "Efficiency &amp; Cost Impact" in r_dash.text or "Efficiency & Cost Impact" in r_dash.text
assert "LLM Calls Avoided" in r_dash.text
assert "LLM Calls Executed" in r_dash.text
assert "How Semantic Caching Helps" in r_dash.text
assert "Where Time Is Spent" in r_dash.text
assert "ROUTING DISTRIBUTION" in r_dash.text
print("[OK] Dashboard HTML contains all required Efficiency & Cost Impact elements.")

# Step 9: Error Telemetry Verification
print("\n[TEST 9] Error Telemetry Verification")
# Bad query: empty string should return 422/400 validation error
r_bad = requests.post(f"{BASE}/api/chat", json={"query": ""})
print(f"Empty Query Status: {r_bad.status_code} (Expected 422 or 400)")
assert r_bad.status_code in (400, 422), f"Expected 400 or 422, got {r_bad.status_code}"

from datetime import datetime, timezone
from backend.app.analytics import analytics_tracker
from backend.app.monitoring.telemetry import TelemetryRecord

err_rec = TelemetryRecord(
    request_id="test-error-id-001",
    timestamp=datetime.now(timezone.utc).isoformat(),
    query="Invalid query test",
    status="error",
    error_stage="llm_generation",
    error_message="Simulated connection timeout",
    total_request_latency_ms=500.0,
    actual_api_cost=0.0
)
analytics_tracker.record_telemetry(err_rec)


r_recent_err = requests.get(f"{BASE}/api/analytics/recent?limit=1").json()
assert r_recent_err[0]["status"] == "error", "Expected error status in recent telemetry"
assert r_recent_err[0]["error_stage"] == "llm_generation"
assert r_recent_err[0]["error_message"] == "Simulated connection timeout"
print("[OK] Error telemetry successfully recorded and verified in recent stream.")


# Step 10: Persistence Verification (SQLite records survive in database)
print("\n[TEST 10] SQLite Telemetry Persistence Verification")
from backend.app.monitoring.storage import telemetry_storage
total_in_db = telemetry_storage.count()
print(f"Telemetry DB Records Count: {total_in_db}")
assert total_in_db == 4, f"Expected 4 records in SQLite DB, got {total_in_db}"

print("\n" + "=" * 60)
print("ALL PHASE 4 TELEMETRY & ANALYTICS TESTS COMPLETED SUCCESSFULLY!")
print("=" * 60)

