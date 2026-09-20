"""Tests for the Ollama HTTP API agent.

The agent is a thin wrapper around the Ollama ``/api/chat`` and ``/api/generate``
endpoints. We mock urllib so no real server is needed.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ai_env.ollama_agent import OllamaCodingAgent


SYSTEM_PROMPT = "You are a test assistant."


def _fake_response(payload: dict, status: int = 200):
    body = json.dumps(payload).encode("utf-8")

    class _Resp:
        def __init__(self):
            self.status = status

        def read(self):
            return body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    return _Resp()


class TestOllamaCodingAgentInit:
    def test_stores_host_port_model(self):
        agent = OllamaCodingAgent(host="h", port=1, default_model="m")
        assert agent.host == "h"
        assert agent.port == 1
        assert agent.default_model == "m"

    def test_default_model_present(self):
        agent = OllamaCodingAgent()
        assert agent.default_model  # not empty


class TestIsReachable:
    def test_true_when_health_endpoint_responds(self):
        with patch(
            "ai_env.ollama_agent.urllib.request.urlopen",
            return_value=_fake_response({}),
        ):
            assert OllamaCodingAgent.is_reachable("h", 1) is True

    def test_false_on_exception(self):
        with patch(
            "ai_env.ollama_agent.urllib.request.urlopen",
            side_effect=OSError("nope"),
        ):
            assert OllamaCodingAgent.is_reachable("h", 1) is False


class TestListModels:
    def test_returns_model_names(self):
        payload = {"models": [{"name": "qwen2.5-coder:7b"}, {"name": "llama3:8b"}]}
        with patch(
            "ai_env.ollama_agent.urllib.request.urlopen",
            return_value=_fake_response(payload),
        ):
            models = OllamaCodingAgent.list_models("h", 1)
        assert models == ["qwen2.5-coder:7b", "llama3:8b"]

    def test_returns_empty_list_when_no_models(self):
        with patch(
            "ai_env.ollama_agent.urllib.request.urlopen",
            return_value=_fake_response({}),
        ):
            assert OllamaCodingAgent.list_models("h", 1) == []


class TestGenerate:
    def test_generate_uses_chat_endpoint(self):
        agent = OllamaCodingAgent(host="h", port=1, default_model="m")
        with patch(
            "ai_env.ollama_agent.urllib.request.urlopen",
            return_value=_fake_response({"message": {"content": "hi"}}),
        ) as urlopen:
            response = agent.generate("hello", max_tokens=64, temperature=0.5)
        assert response == "hi"
        # Inspect what we sent
        req = urlopen.call_args.args[0]
        assert req.full_url == "http://h:1/api/chat"
        body = json.loads(req.data.decode("utf-8"))
        assert body["model"] == "m"
        assert body["stream"] is False
        assert body["messages"][-1]["content"] == "hello"
        assert body["options"]["temperature"] == 0.5
        assert body["options"]["num_predict"] == 64

    def test_generate_appends_to_history(self):
        agent = OllamaCodingAgent(host="h", port=1, default_model="m")
        with patch(
            "ai_env.ollama_agent.urllib.request.urlopen",
            return_value=_fake_response({"message": {"content": "answer"}}),
        ):
            agent.generate("first question")
            agent.generate("follow up")
        assert len(agent.conversation_history) == 4
        assert agent.conversation_history[0]["role"] == "user"
        assert agent.conversation_history[1]["role"] == "assistant"
        assert agent.conversation_history[2]["role"] == "user"
        assert agent.conversation_history[3]["role"] == "assistant"

    def test_generate_raises_on_http_error(self):
        agent = OllamaCodingAgent(host="h", port=1, default_model="m")
        with patch(
            "ai_env.ollama_agent.urllib.request.urlopen",
            return_value=_fake_response({}, status=500),
        ):
            with pytest.raises(RuntimeError):
                agent.generate("hi")


class TestStatusAndHistory:
    def test_status_includes_metadata(self):
        agent = OllamaCodingAgent(host="h", port=1, default_model="m")
        agent.set_workspace_root("/tmp")
        with patch(
            "ai_env.ollama_agent.urllib.request.urlopen",
            return_value=_fake_response({"message": {"content": "ok"}}),
        ):
            agent.generate("q")
        status = agent.get_status()
        assert status["model_name"] == "m"
        assert status["loaded"] is True
        assert status["conversation_turns"] == 2
        assert status["total_requests"] == 1
        from pathlib import Path
        assert status["workspace_root"] == str(Path("/tmp").resolve())
        assert status["last_response_source"] == "ollama"

    def test_reset_history_clears_state(self):
        agent = OllamaCodingAgent()
        agent.conversation_history.append({"role": "user", "content": "x"})
        agent.tokens_used = 99
        agent.total_requests = 3
        agent.reset_history()
        assert agent.conversation_history == []
        assert agent.tokens_used == 0
        assert agent.total_requests == 0
