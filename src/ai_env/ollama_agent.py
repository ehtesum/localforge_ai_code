"""Coding agent backed by a local Ollama server (HTTP API).

This is a lightweight counterpart to :mod:`ai_env.agent` for environments where
we don't want to ship torch / transformers (e.g. the Docker image). It talks to
Ollama's ``/api/chat`` and ``/api/tags`` endpoints using only the standard
library.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, ClassVar, List, Optional

SYSTEM_PROMPT = (
    "You are an expert AI coding assistant. You help write, debug, and explain "
    "code. CRITICAL: You must write down your thinking process before providing "
    "the final answer. Wrap your thinking process in <thought>...</thought> tags. "
    "After the closing </thought> tag, provide your final response to the user. "
    "Keep the final response clean, well-structured, and concise."
)


class OllamaCodingAgent:
    """Conversation-aware coding agent that delegates inference to Ollama."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        default_model: str = "qwen2.5-coder:7b",
    ) -> None:
        self.host = host
        self.port = port
        self.default_model = default_model
        self.conversation_history: List[dict] = []
        self.tokens_used: int = 0
        self.total_requests: int = 0
        self.workspace_root: Optional[Path] = None
        self.last_response_source: str = "ollama"
        self.last_response_text: str = ""

    # ------------------------------------------------------------------ HTTP

    @property
    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _request(self, path: str, payload: Optional[dict] = None, timeout: int = 120) -> dict:
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status >= 400:
                    raise RuntimeError(f"Ollama HTTP {response.status}")
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Ollama at {url}: {exc}") from exc

    # ----------------------------------------------------------- Introspection

    @classmethod
    def is_reachable(cls, host: str, port: int, timeout: float = 1.5) -> bool:
        try:
            with urllib.request.urlopen(
                f"http://{host}:{port}/api/version", timeout=timeout
            ):
                return True
        except (urllib.error.URLError, OSError):
            return False

    @classmethod
    def list_models(cls, host: str, port: int, timeout: int = 5) -> List[str]:
        url = f"http://{host}:{port}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return []
        return [m.get("name", "") for m in payload.get("models", []) if m.get("name")]

    def available_models(self) -> List[str]:
        return self.list_models(self.host, self.port)

    # ------------------------------------------------------------------ Chat

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        model: Optional[str] = None,
    ) -> str:
        """Send ``prompt`` to Ollama and return the model's reply."""
        model_name = model or self.default_model
        self.conversation_history.append({"role": "user", "content": prompt})

        messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.conversation_history)

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }

        response = self._request("/api/chat", payload)
        content = (response.get("message") or {}).get("content", "")

        self.conversation_history.append({"role": "assistant", "content": content})
        self.total_requests += 1
        # Ollama returns ``eval_count`` when not streaming
        self.tokens_used += int(response.get("eval_count") or 0)
        self.last_response_text = content
        self.last_response_source = "ollama"
        return content

    def complete(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.2,
        model: Optional[str] = None,
    ) -> str:
        """Call Ollama's /api/generate directly for single-turn code completion."""
        model_name = model or self.default_model
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        response = self._request("/api/generate", payload)
        return response.get("response", "")

    # ----------------------------------------------------------- Convenience

    def get_status(self) -> dict:
        return {
            "model_name": self.default_model,
            "device": "ollama",
            "loaded": True,
            "tokens_used": self.tokens_used,
            "conversation_turns": len(self.conversation_history),
            "total_requests": self.total_requests,
            "workspace_root": str(self.workspace_root or Path.cwd()),
            "last_response_source": self.last_response_source,
            "last_response_text": self.last_response_text,
            "available_models": self.available_models(),
        }

    def set_workspace_root(self, root_path) -> None:
        self.workspace_root = Path(root_path).resolve()

    def reset_history(self) -> None:
        self.conversation_history = []
        self.tokens_used = 0
        self.total_requests = 0

    def load_model(self) -> None:  # pragma: no cover - no-op for HTTP backend
        return None
