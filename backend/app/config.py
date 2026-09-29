import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Base Directory paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

# Ensure essential directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Application configuration settings."""

    # Project metadata
    PROJECT_NAME: str = "RAG Cost Optimizer - Phase 1"
    PROJECT_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Storage Paths
    DOCUMENTS_PATH: Path = DOCUMENTS_DIR
    CHROMA_DB_PATH: Path = CHROMA_PERSIST_DIR
    CHROMA_COLLECTION_NAME: str = "rag_cost_optimizer_documents"

    # Embedding Settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 60

    # Retrieval Settings
    TOP_K_RETRIEVAL: int = 3

    # Semantic Cache Settings (Phase 2)
    SEMANTIC_CACHE_ENABLED: bool = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
    SEMANTIC_CACHE_THRESHOLD: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.90"))
    SEMANTIC_CACHE_COLLECTION_NAME: str = "semantic_cache"

    # Cost-Aware Intelligent Routing Settings (Phase 3)
    COST_ROUTING_ENABLED: bool = os.getenv("COST_ROUTING_ENABLED", "true").lower() in ("true", "1", "yes")
    SMALL_MODEL: str = os.getenv("SMALL_MODEL", "llama3.2:3b")
    LARGE_MODEL: str = os.getenv("LARGE_MODEL", "mistral:7b")
    SIMPLE_THRESHOLD: int = int(os.getenv("SIMPLE_THRESHOLD", "3"))
    COMPLEX_THRESHOLD: int = int(os.getenv("COMPLEX_THRESHOLD", "7"))

    # LLM Settings (Local Ollama)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180.0"))
    TEMPERATURE: float = 0.1

    # Strict Context Prompt Template
    SYSTEM_PROMPT: str = (
        "You are an accurate, honest AI assistant for question answering.\n"
        "Your task is to answer the user's question using ONLY the provided retrieved context.\n"
        "STRICT RULES:\n"
        "1. Base your answer strictly on the provided context.\n"
        "2. If the answer is not available or cannot be deduced directly from the context, clearly state: "
        "'I am sorry, but the answer is not available in the provided documents.'\n"
        "3. Do not assume or extrapolate beyond the provided text.\n"
        "4. Do not use external knowledge or invent facts.\n"
        "5. Keep your response concise, clear, and professional."
    )

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        extra = "allow"


settings = Settings()
