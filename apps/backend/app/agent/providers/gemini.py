import os
import logging
import google.generativeai as genai

from typing import Any, Dict
from fastapi.concurrency import run_in_threadpool

from ..exceptions import ProviderError
from .base import Provider, EmbeddingProvider

logger = logging.getLogger(__name__)


class GeminiProvider(Provider):
    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-flash"):
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderError("Gemini API key is missing. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        
        genai.configure(api_key=api_key)
        self.model_name = model
        self._model = genai.GenerativeModel(model)

    def _generate_sync(self, prompt: str, options: Dict[str, Any]) -> str:
        """
        Generate a response from the Gemini model.
        """
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=options.get("temperature", 0.0),
                top_p=options.get("top_p", 0.9),
                top_k=options.get("top_k", 40),
                max_output_tokens=options.get("max_tokens", 8192),
            )
            
            response = self._model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if not response.text:
                raise ProviderError("Gemini - Empty response received")
                
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini sync error: {e}")
            raise ProviderError(f"Gemini - Error generating response: {e}")

    async def __call__(self, prompt: str, **generation_args: Any) -> str:
        opts = {
            "temperature": generation_args.get("temperature", 0.0),
            "top_p": generation_args.get("top_p", 0.9),
            "top_k": generation_args.get("top_k", 40),
            "max_tokens": generation_args.get("max_length", 8192),
        }
        return await run_in_threadpool(self._generate_sync, prompt, opts)


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str | None = None,
        embedding_model: str = "gemini-embedding-001",
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderError("Gemini API key is missing. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        
        genai.configure(api_key=api_key)
        self._model = embedding_model

    def _embed_sync(self, text: str) -> list[float]:
        """
        Generate an embedding for the given text using Gemini.
        """
        try:
            result = genai.embed_content(
                model=self._model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            raise ProviderError(f"Gemini - Error generating embedding: {e}")

    async def embed(self, text: str) -> list[float]:
        """
        Generate an embedding for the given text.
        """
        return await run_in_threadpool(self._embed_sync, text)