import sys
import json
from backend.app.config import settings
from backend.app.llm import ollama_client
from backend.app.router.query_router import query_router
from backend.app.schemas import SourceChunk

print("--- DIRECT INFERENCE TEST WITH mistral:7b ---")
try:
    chunks = [
        SourceChunk(
            chunk_id="chunk_test_1",
            content="Database normalization organizes tables to reduce redundancy and improve data integrity.",
            source="test.txt",
            score=0.95
        )
    ]
    response = ollama_client.generate_answer(
        query="What is database normalization?",
        context_chunks=chunks,
        model="mistral:7b"
    )
    print(f"Direct Mistral generation SUCCESS! Length: {len(response)} chars")
    print(f"Response Preview:\n{response}")
except Exception as e:
    import traceback
    print(f"Direct Mistral generation FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
