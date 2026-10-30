"""A tiny provider-agnostic LLM client.

Works with any OpenAI-compatible endpoint (Google AI Studio, Groq, OpenRouter,
Ollama...). Switching provider = changing three lines in .env, no code changes.
"""
import os
import time

from dotenv import load_dotenv
from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError

load_dotenv()


class LLMClient:
    def __init__(self, api_key=None, base_url=None, model=None, max_retries=4):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL")
        self.max_retries = max_retries

        missing = [n for n, v in [("LLM_API_KEY", self.api_key),
                                  ("LLM_BASE_URL", self.base_url),
                                  ("LLM_MODEL", self.model)] if not v]
        if missing:
            raise RuntimeError(f"Missing in .env: {', '.join(missing)}")

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60,      # give up after 60s instead of 10 minutes
            max_retries=0,   # we do our own retries below, and print them
        )

    def complete(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Send one system + user message and return the reply text.

        Free tiers are flaky: 429 means you hit a rate limit, 503 means the
        provider is overloaded. Both are temporary, so wait and retry with
        exponential backoff instead of crashing.
        """
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return response.choices[0].message.content or ""
            except (RateLimitError, InternalServerError, APIConnectionError) as e:
                if attempt == self.max_retries - 1:
                    raise
                wait = 10 * 2 ** attempt
                print(f"{type(e).__name__}, waiting {wait}s (attempt {attempt + 1}/{self.max_retries})...")
                time.sleep(wait)
        return ""
