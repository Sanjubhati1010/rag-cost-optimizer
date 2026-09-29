import time
import httpx
from backend.app.rag_service import rag_service

q = "Compare 1NF, 2NF, 3NF and BCNF with detailed examples. Explain functional dependencies, candidate keys, partial dependency and transitive dependency, and discuss when each normalization level should be applied in a real database design."

emb = rag_service.embedding_service.get_embedding(q)
chunks = rag_service.vector_store.search_similar(query_embedding=emb, top_k=3)
print(f"Retrieved {len(chunks)} chunks.")

prompt = rag_service.llm_client.format_prompt(q, chunks)
print(f"Formatted prompt length: {len(prompt)} chars ({len(prompt.split())} words)")

payload = {
    "model": "mistral:7b",
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.1,
        "top_p": 0.9,
        "num_predict": 256,
    }
}

t0 = time.time()
print("Starting Ollama generate with num_predict=256...")
with httpx.Client(timeout=300.0) as client:
    res = client.post("http://127.0.0.1:11434/api/generate", json=payload)
    t1 = time.time()
    print(f"Status: {res.status_code}, Time: {t1 - t0:.2f}s")
    if res.status_code == 200:
        data = res.json()
        ans = data.get("response", "")
        print(f"Response ({len(ans)} chars):\n{ans}")
    else:
        print("Error:", res.text)
