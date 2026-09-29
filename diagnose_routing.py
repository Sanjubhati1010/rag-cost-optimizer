import sys
import json
from backend.app.config import settings
from backend.app.llm import ollama_client
from backend.app.router.query_router import query_router
from backend.app.schemas import SourceChunk

print("--- 1. CONFIGURATION ---")
print(f"SMALL_MODEL: {settings.SMALL_MODEL}")
print(f"LARGE_MODEL: {settings.LARGE_MODEL}")
print(f"COST_ROUTING_ENABLED: {settings.COST_ROUTING_ENABLED}")

print("\n--- 2. OLLAMA HEALTH & DETECTED MODELS ---")
health = ollama_client.check_health()
print(f"Health check result: {health}")

print("\n--- 3. MODEL AVAILABILITY CHECKS ---")
print(f"is_model_available('{settings.SMALL_MODEL}'): {ollama_client.is_model_available(settings.SMALL_MODEL)}")
print(f"is_model_available('{settings.LARGE_MODEL}'): {ollama_client.is_model_available(settings.LARGE_MODEL)}")
print(f"is_model_available('mistral:7b'): {ollama_client.is_model_available('mistral:7b')}")

print("\n--- 4. DIRECT INFERENCE TEST WITH mistral:7b ---")
try:
    chunks = [SourceChunk(content="Database normalization organizes tables to reduce redundancy.", source="test.txt", score=0.95)]
    response = ollama_client.generate_answer(
        query="What is normalization?",
        context_chunks=chunks,
        model="mistral:7b"
    )
    print(f"Direct Mistral generation SUCCESS! Length: {len(response)} chars")
    print(f"Preview: {response[:150]}...")
except Exception as e:
    print(f"Direct Mistral generation FAILED: {type(e).__name__}: {e}")

print("\n--- 5. ROUTING DECISION TEST ---")
complex_q = "Compare 1NF, 2NF, 3NF and BCNF with examples, explain functional dependencies and discuss when each normalization level should be used."
decision = query_router.route_query(complex_q)
print("Routing decision output:")
print(json.dumps(decision.model_dump(), indent=2))
