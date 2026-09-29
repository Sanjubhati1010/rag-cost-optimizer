from typing import List
from sentence_transformers import SentenceTransformer
from backend.app.config import settings
from backend.app.logging_config import logger

class EmbeddingService:
    """Manages SentenceTransformer embedding model loading and inference."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._model = None
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-loads the SentenceTransformer embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            logger.info(f"Successfully loaded embedding model: {settings.EMBEDDING_MODEL_NAME}")
        return self._model

    def get_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for a single string.

        Args:
            text: Input text.

        Returns:
            List of floats representing the embedding vector.
        """
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a batch of strings.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()


embedding_service = EmbeddingService()
