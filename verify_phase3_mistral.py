import requests
import json

BASE = "http://127.0.0.1:8000"

print("=== STEP 1: CLEARING CACHE ===")
rc = requests.delete(f"{BASE}/api/cache")
print("Cache clear response:", rc.status_code, rc.json())

print("\n=== STEP 2: RUNNING COMPLEX QUERY ===")
complex_query = "Compare 1NF, 2NF, 3NF and BCNF with detailed examples. Explain functional dependencies, candidate keys, partial dependency and transitive dependency, and discuss when each normalization level should be applied in a real database design."
payload = {"query": complex_query}

r_complex = requests.post(f"{BASE}/api/chat", json=payload, timeout=180)
print(f"Complex Query HTTP Status: {r_complex.status_code}")
res = r_complex.json()
print("Complex Query Response JSON:")
print(json.dumps(res, indent=2))

# Assertions
assert res["cache_hit"] is False, "Expected cache_hit=False on fresh query"
assert res["complexity_level"] == "complex", f"Expected complexity_level='complex', got {res['complexity_level']}"
assert res["complexity_score"] >= 7, f"Expected complexity_score >= 7, got {res['complexity_score']}"
assert res["selected_model"] == "mistral:7b", f"Expected selected_model='mistral:7b', got {res['selected_model']}"
assert "fallback" not in res["routing_reason"].lower(), f"Unexpected fallback in reason: {res['routing_reason']}"

print("\n=== STEP 3: RUNNING SIMPLE QUERY ===")
simple_query = "What is DBMS?"
r_simple = requests.post(f"{BASE}/api/chat", json={"query": simple_query})
print(f"Simple Query HTTP Status: {r_simple.status_code}")
res_s = r_simple.json()
print("Simple Query Response JSON:")
print(json.dumps(res_s, indent=2))

assert res_s["cache_hit"] is False, "Expected cache_hit=False"
assert res_s["complexity_level"] == "simple", f"Expected simple, got {res_s['complexity_level']}"
assert res_s["selected_model"] == "llama3.2:3b", f"Expected llama3.2:3b, got {res_s['selected_model']}"

print("\n=== STEP 4: REPEAT SIMPLE QUERY (CACHE HIT) ===")
r_hit = requests.post(f"{BASE}/api/chat", json={"query": simple_query})
res_h = r_hit.json()
print("Cache Hit Response JSON:")
print(json.dumps(res_h, indent=2))

assert res_h["cache_hit"] is True, "Expected cache_hit=True"
assert res_h["complexity_level"] == "not_analyzed", "Expected not_analyzed"
assert res_h["selected_model"] == "not_used", "Expected not_used"
assert res_h["routing_reason"] == "Cache hit; routing skipped"

print("\n=== STEP 5: VERIFY ANALYTICS ===")
r_a = requests.get(f"{BASE}/api/analytics")
res_a = r_a.json()
print("Analytics JSON:")
print(json.dumps(res_a, indent=2))

assert res_a["total_requests"] == 3
assert res_a["cache_hits"] == 1
assert res_a["cache_misses"] == 2
assert res_a["small_model_requests"] == 1
assert res_a["large_model_requests"] == 1
assert res_a["simple_queries"] == 1
assert res_a["complex_queries"] == 1
assert res_a["actual_api_cost"] == 0.0

print("\n==================================================")
print(">>> ALL VERIFICATION TESTS PASSED SUCCESSFULLY! <<<")
print("==================================================")
