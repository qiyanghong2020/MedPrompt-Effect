from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter


class OpenRouterError(RuntimeError):
    pass


class OpenRouterRetryableError(OpenRouterError):
    pass


class OpenRouterNonRetryableError(OpenRouterError):
    pass


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.0
    max_tokens: int = 64
    top_p: float | None = None
    stop: list[str] | None = None

    def cache_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path = "cache/openrouter",
        http_referer: str | None = None,
        x_title: str | None = None,
        timeout_s: int = 90,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip(
            "/"
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.http_referer = http_referer or os.environ.get("OPENROUTER_HTTP_REFERER")
        self.x_title = x_title or os.environ.get("OPENROUTER_X_TITLE")
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title
        return headers

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, key: str, value: dict[str, Any]) -> None:
        path = self._cache_path(key)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    @retry(
        retry=retry_if_exception_type((requests.RequestException, OpenRouterRetryableError)),
        wait=wait_exponential_jitter(initial=0.8, max=10),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def chat(self, req: ChatRequest) -> dict[str, Any]:
        key = req.cache_key()
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": req.messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if req.top_p is not None:
            payload["top_p"] = req.top_p
        if req.stop is not None:
            payload["stop"] = req.stop

        resp = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=self.timeout_s)
        if resp.status_code != 200:
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text[:1000]}

            # Retry on rate limits and transient upstream/provider failures.
            if resp.status_code in (408, 409, 425, 429) or resp.status_code >= 500:
                raise OpenRouterRetryableError(f"HTTP {resp.status_code}: {body}")

            # Treat other 4xx as non-retryable (bad request, auth, etc.).
            raise OpenRouterNonRetryableError(f"HTTP {resp.status_code}: {body}")

        data = resp.json()
        self._write_cache(key, data)
        time.sleep(0.02)
        return data

    def list_models(self) -> list[dict[str, Any]]:
        url = f"{self.base_url}/models"
        resp = requests.get(url, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json().get("data", [])
