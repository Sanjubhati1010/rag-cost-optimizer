import requests
import json

BASE = "http://127.0.0.1:8000"

print("=== RESETTING CACHE ===")
requests.delete(f"{BASE}/api/cache")

print("\n=== TEST 1: Simple query (What is DBMS?) ===")
q1 = "What is DBMS?"
r1 = requests.post(f"{BASE}/api/chat", json={"query": q1})
print("T1 Status:", r1.status_code)
d1 = r1.json()
print("T1 Response:", json.dumps(d1, indent=2))
assert d1["cache_hit"] is False, "Expected cache miss"
assert d1["complexity_level"] == "simple", f"Expected simple, got {d1['complexity_level']}"
assert d1["complexity_score"] <= 3, f"Expected score <= 3, got {d1['complexity_score']}"
assert "3b" in d1["selected_model"], "Expected small model"

print("\n=== TEST 2: Medium/Simple query (Explain normalization in DBMS.) ===")
q2 = "Explain normalization in DBMS."
r2 = requests.post(f"{BASE}/api/chat", json={"query": q2})
print("T2 Status:", r2.status_code)
d2 = r2.json()
print("T2 Response:", json.dumps(d2, indent=2))
assert d2["complexity_level"] in ("simple", "medium"), f"Expected simple/medium, got {d2['complexity_level']}"
assert "3b" in d2["selected_model"], "Expected small model"

print("\n=== TEST 3: Complex query ===")
q3 = "Compare 1NF, 2NF, 3NF and BCNF with examples, explain functional dependencies and discuss when each normalization level should be used."
r3 = requests.post(f"{BASE}/api/chat", json={"query": q3})
print("T3 Status:", r3.status_code)
d3 = r3.json()
print("T3 Response:", json.dumps(d3, indent=2))
assert d3["complexity_level"] == "complex", f"Expected complex, got {d3['complexity_level']}"
assert d3["complexity_score"] >= 7, f"Expected score >= 7, got {d3['complexity_score']}"
assert d3["selected_model"] == "mistral:7b", f"Expected mistral:7b, got {d3['selected_model']}"
assert "fallback" not in d3["routing_reason"].lower(), f"Unexpected fallback in routing reason: {d3['routing_reason']}"

print("\n=== TEST 4: Repeat TEST 1 (Expect Cache HIT, router bypassed) ===")
r4 = requests.post(f"{BASE}/api/chat", json={"query": q1})
print("T4 Status:", r4.status_code)
d4 = r4.json()
print("T4 Response:", json.dumps(d4, indent=2))
assert d4["cache_hit"] is True, "Expected cache hit"
assert d4["complexity_score"] is None, "Expected complexity_score to be None on cache hit"
assert d4["complexity_level"] == "not_analyzed", "Expected complexity_level not_analyzed"
assert d4["selected_model"] == "not_used", "Expected selected_model not_used"
assert d4["routing_reason"] == "Cache hit; routing skipped"

print("\n=== TEST 5: GET /api/analytics ===")
ra = requests.get(f"{BASE}/api/analytics")
print("Analytics Status:", ra.status_code)
da = ra.json()
print("Analytics:", json.dumps(da, indent=2))
assert da["total_requests"] == 4
assert da["cache_hits"] == 1
assert da["cache_misses"] == 3
assert da["actual_api_cost"] == 0.0

print("\nALL PHASE 3 ROUTING & ANALYTICS TESTS PASSED!")
