"""Tests for launch.py."""
from __future__ import annotations

import argparse
import socket
from unittest.mock import patch

import pytest

import launch


def _args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        host="127.0.0.1",
        port=7861,
        root="/tmp",
        no_browser=True,
        backend="auto",
        ollama_host="127.0.0.1",
        ollama_port=11434,
        ollama_model="m",
        wait_for_ollama=0,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestEnsureSrcOnPath:
    def test_inserts_src_into_sys_path(self, tmp_path, monkeypatch):
        # Run the function as if launch.py were inside tmp_path
        fake_launch = tmp_path / "launch.py"
        fake_launch.write_text("", encoding="utf-8")
        (tmp_path / "src").mkdir()
        monkeypatch.setattr(launch, "__file__", str(fake_launch))
        monkeypatch.setattr(launch.sys, "path", [])

        launch._ensure_src_on_path()
        assert str(tmp_path / "src") in launch.sys.path

    def test_raises_when_src_missing(self, tmp_path, monkeypatch):
        fake_launch = tmp_path / "launch.py"
        fake_launch.write_text("", encoding="utf-8")
        monkeypatch.setattr(launch, "__file__", str(fake_launch))
        with pytest.raises(SystemExit):
            launch._ensure_src_on_path()


class TestOllamaReachable:
    def test_true_on_successful_connect(self):
        with patch.object(socket, "create_connection") as cc:
            assert launch._ollama_reachable("h", 1) is True
            cc.assert_called_once()

    def test_false_on_oserror(self):
        with patch.object(socket, "create_connection", side_effect=OSError("nope")):
            assert launch._ollama_reachable("h", 1) is False


class TestResolveBackend:
    def test_explicit_backend_returned(self):
        assert launch._resolve_backend(_args(backend="ollama")) == "ollama"
        assert launch._resolve_backend(_args(backend="local")) == "local"

    def test_auto_picks_ollama_when_reachable(self):
        with patch.object(launch, "_ollama_reachable", return_value=True):
            assert launch._resolve_backend(_args(backend="auto")) == "ollama"

    def test_auto_falls_back_to_local(self):
        with patch.object(launch, "_ollama_reachable", return_value=False):
            assert launch._resolve_backend(_args(backend="auto")) == "local"


class TestBuildAgent:
    def test_ollama_backend_returns_ollama_agent(self):
        from ai_env.ollama_agent import OllamaCodingAgent

        with patch.dict("sys.modules", {"ai_env.ollama_agent": pytest.importorskip("ai_env.ollama_agent")}):
            agent = launch._build_agent("ollama", _args(ollama_host="h", ollama_port=2, ollama_model="m"))
        assert isinstance(agent, OllamaCodingAgent)
        assert agent.host == "h"
        assert agent.port == 2
        assert agent.default_model == "m"

    def test_local_backend_returns_local_agent(self):
        from ai_env.agent import CodingAgent
        agent = launch._build_agent("local", _args())
        assert isinstance(agent, CodingAgent)


class TestWaitForOllama:
    def test_returns_true_when_endpoint_responds(self):
        # First call to ``_ollama_reachable`` returns False (startup),
        # second returns True and the API responds.
        with patch.object(launch, "_ollama_reachable", side_effect=[False, True]), \
             patch("launch.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.status = 200
            assert launch._wait_for_ollama("h", 1, timeout=5) is True

    def test_returns_false_on_timeout(self, monkeypatch):
        monkeypatch.setattr(launch.time, "time", lambda: 0)  # always deadline-expired
        with patch.object(launch, "_ollama_reachable", return_value=False):
            assert launch._wait_for_ollama("h", 1, timeout=0) is False
