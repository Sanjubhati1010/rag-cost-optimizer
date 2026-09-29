import httpx
from typing import List, Optional, Dict, Any, Tuple
from backend.app.config import settings

from backend.app.logging_config import logger
from backend.app.schemas import SourceChunk


class OllamaLLMClient:
    """Client for generating answers using local Ollama LLM."""

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.OLLAMA_MODEL,
        timeout: float = settings.OLLAMA_TIMEOUT_SECONDS,
        temperature: float = settings.TEMPERATURE
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    def check_health(self) -> Dict[str, Any]:
        """Checks if Ollama server is running and accessible."""
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    models = [m.get("name") for m in res.json().get("models", [])]
                    # Check if our model (e.g. llama3.2:3b or llama3.2) is present
                    model_found = any(self.model in m or m.startswith(self.model.split(":")[0]) for m in models)
                    return {
                        "connected": True,
                        "model_available": model_found,
                        "available_models": models
                    }
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return {
                "connected": False,
                "model_available": False,
                "available_models": [],
                "error": str(e)
            }
        return {"connected": False, "model_available": False, "available_models": []}

    def format_prompt(self, query: str, context_chunks: List[SourceChunk]) -> str:
        """Constructs a strict context-bound prompt for the LLM."""
        if not context_chunks:
            context_text = "NO CONTEXT AVAILABLE."
        else:
            context_blocks = []
            for i, chunk in enumerate(context_chunks, start=1):
                block = f"[Document Chunk {i} | Source: {chunk.source}]\n{chunk.content}"
                context_blocks.append(block)
            context_text = "\n\n".join(context_blocks)

        prompt = (
            f"{settings.SYSTEM_PROMPT}\n\n"
            f"--- START RETRIEVED CONTEXT ---\n"
            f"{context_text}\n"
            f"--- END RETRIEVED CONTEXT ---\n\n"
            f"User Question: {query}\n\n"
            f"Provide your answer using ONLY the context above. If the answer is not present in the context, "
            f"say 'I am sorry, but the answer is not available in the provided documents.':\n"
            f"Answer:"
        )
        return prompt

    def is_model_available(self, model_name: str) -> bool:
        """Checks whether a specific model is downloaded and available in Ollama."""
        health = self.check_health()
        if not health.get("connected"):
            return False
        available = health.get("available_models", [])
        base_name = model_name.split(":")[0]
        return any(model_name == m or m.startswith(f"{base_name}:") or m == base_name for m in available)

    def generate_answer_with_telemetry(
        self,
        query: str,
        context_chunks: List[SourceChunk],
        model: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Calls the local Ollama LLM to generate an answer and captures timing and token metadata.

        Args:
            query: The user's question.
            context_chunks: Top retrieved document chunks.
            model: Optional model name to override default.

        Returns:
            Tuple of (answer_text, telemetry_dict)
        """
        import time

        target_model = model if model else self.model
        prompt = self.format_prompt(query, context_chunks)
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                "num_predict": 384,
            }
        }

        url = f"{self.base_url}/api/generate"
        logger.info(f"Sending prompt to Ollama model '{target_model}' at {url}...")
        start_t = time.perf_counter()

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                
                if response.status_code != 200:
                    error_msg = f"Ollama returned HTTP status {response.status_code}: {response.text}"
                    logger.warning(error_msg)
                    # If large model failed, try fallback to default small model
                    if target_model != settings.SMALL_MODEL:
                        logger.warning(
                            f"Model '{target_model}' failed with status {response.status_code}. Attempting fallback to '{settings.SMALL_MODEL}'..."
                        )
                        payload["model"] = settings.SMALL_MODEL
                        fb_start = time.perf_counter()
                        fallback_resp = client.post(url, json=payload)
                        fb_end = time.perf_counter()
                        if fallback_resp.status_code == 200:
                            fb_json = fallback_resp.json()
                            fb_text = fb_json.get("response", "").strip()
                            prompt_eval_count = fb_json.get("prompt_eval_count")
                            eval_count = fb_json.get("eval_count")
                            load_dur = fb_json.get("load_duration")
                            prompt_eval_dur = fb_json.get("prompt_eval_duration")
                            eval_dur = fb_json.get("eval_duration")
                            total_tokens = (prompt_eval_count + eval_count) if (prompt_eval_count is not None and eval_count is not None) else None

                            meta = {
                                "model": settings.SMALL_MODEL,
                                "generation_latency_ms": round((fb_end - fb_start) * 1000, 2),
                                "input_tokens": prompt_eval_count,
                                "output_tokens": eval_count,
                                "total_tokens": total_tokens,
                                "load_duration_ms": round(load_dur / 1e6, 2) if load_dur is not None else None,
                                "prompt_eval_duration_ms": round(prompt_eval_dur / 1e6, 2) if prompt_eval_dur is not None else None,
                                "eval_duration_ms": round(eval_dur / 1e6, 2) if eval_dur is not None else None,
                            }
                            return fb_text, meta
                    raise RuntimeError(error_msg)

                end_t = time.perf_counter()
                result = response.json()
                raw_response = result.get("response", "").strip()
                gen_latency_ms = round((end_t - start_t) * 1000, 2)

                prompt_eval_count = result.get("prompt_eval_count")
                eval_count = result.get("eval_count")
                load_dur = result.get("load_duration")
                prompt_eval_dur = result.get("prompt_eval_duration")
                eval_dur = result.get("eval_duration")
                total_tokens = (prompt_eval_count + eval_count) if (prompt_eval_count is not None and eval_count is not None) else None

                meta = {
                    "model": target_model,
                    "generation_latency_ms": gen_latency_ms,
                    "input_tokens": prompt_eval_count,
                    "output_tokens": eval_count,
                    "total_tokens": total_tokens,
                    "load_duration_ms": round(load_dur / 1e6, 2) if load_dur is not None else None,
                    "prompt_eval_duration_ms": round(prompt_eval_dur / 1e6, 2) if prompt_eval_dur is not None else None,
                    "eval_duration_ms": round(eval_dur / 1e6, 2) if eval_dur is not None else None,
                }
                logger.info(
                    f"Ollama generated response successfully with model '{target_model}' "
                    f"({len(raw_response)} chars, {gen_latency_ms}ms, tokens={total_tokens})."
                )
                return raw_response, meta

        except httpx.ConnectError as e:
            logger.error(f"Could not connect to Ollama at {self.base_url}. Is the Ollama service running? Error: {e}")
            raise ConnectionError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                "Please verify that Ollama is running (`ollama serve` or Ollama app)."
            ) from e
        except httpx.TimeoutException as e:
            logger.error(f"Ollama request timed out after {self.timeout}s for model '{target_model}': {e}")
            if target_model != settings.SMALL_MODEL:
                logger.warning(
                    f"Model '{target_model}' timed out. Attempting graceful fallback to '{settings.SMALL_MODEL}'..."
                )
                try:
                    return self.generate_answer_with_telemetry(query, context_chunks, model=settings.SMALL_MODEL)
                except Exception as fb_err:
                    logger.error(f"Fallback to '{settings.SMALL_MODEL}' also failed: {fb_err}")
            raise TimeoutError(f"Ollama request timed out after {self.timeout} seconds.") from e
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama model '{target_model}': {e}")
            if target_model != settings.SMALL_MODEL:
                logger.warning(
                    f"Model '{target_model}' encountered error ({e}). Attempting graceful fallback to '{settings.SMALL_MODEL}'..."
                )
                try:
                    return self.generate_answer_with_telemetry(query, context_chunks, model=settings.SMALL_MODEL)
                except Exception as fb_err:
                    logger.error(f"Fallback to '{settings.SMALL_MODEL}' also failed: {fb_err}")
            raise

    def generate_answer(
        self,
        query: str,
        context_chunks: List[SourceChunk],
        model: Optional[str] = None
    ) -> str:
        """Calls the local Ollama LLM to generate an answer based strictly on retrieved context."""
        answer, _ = self.generate_answer_with_telemetry(query, context_chunks, model)
        return answer


ollama_client = OllamaLLMClient()

