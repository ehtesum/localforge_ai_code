"""Tests for src/ai_env/ollama.py."""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from ai_env import ollama


class TestCheckInstalled:
    def test_true_when_executable_found(self):
        with patch.object(ollama, "find_executable", return_value="/usr/bin/ollama"):
            assert ollama.check_installed() is True

    def test_false_when_missing(self):
        with patch.object(ollama, "find_executable", return_value=None):
            assert ollama.check_installed() is False


class TestOllamaCommand:
    def test_builds_command_from_path(self):
        with patch.object(ollama, "find_executable", return_value="/usr/bin/ollama"):
            cmd = ollama._ollama_command()
        assert cmd == ["/usr/bin/ollama"]

    def test_raises_when_missing(self):
        with patch.object(ollama, "find_executable", return_value=None):
            with pytest.raises(FileNotFoundError):
                ollama._ollama_command()


class TestServe:
    def test_serve_starts_subprocess(self):
        fake_proc = type("P", (), {"pid": 1234})()
        with patch.object(ollama, "find_executable", return_value="/usr/bin/ollama"), \
             patch.object(ollama.subprocess, "Popen", return_value=fake_proc) as popen_mock:
            pid = ollama.serve()
        assert pid == 1234
        args, kwargs = popen_mock.call_args
        # First arg is the command list
        assert args[0][0] == "/usr/bin/ollama"
        assert "serve" in args[0]


class TestPullModel:
    def test_pull_runs_command(self):
        with patch.object(ollama, "find_executable", return_value="/usr/bin/ollama"), \
             patch.object(ollama, "run_process", return_value=(0, "ok", "")):
            code = ollama.pull_model("qwen2.5-coder:7b")
        assert code == 0

    def test_pull_raises_on_failure(self):
        with patch.object(ollama, "find_executable", return_value="/usr/bin/ollama"), \
             patch.object(ollama, "run_process", return_value=(1, "", "bad")):
            with pytest.raises(RuntimeError):
                ollama.pull_model("qwen2.5-coder:7b")


class TestListModels:
    def test_returns_stdout(self):
        with patch.object(ollama, "find_executable", return_value="/usr/bin/ollama"), \
             patch.object(ollama, "run_process", return_value=(0, "model-list", "")):
            out = ollama.list_models()
        assert out == "model-list"

    def test_raises_on_failure(self):
        with patch.object(ollama, "find_executable", return_value="/usr/bin/ollama"), \
             patch.object(ollama, "run_process", return_value=(1, "", "nope")):
            with pytest.raises(RuntimeError):
                ollama.list_models()


class TestRemoveModel:
    def test_remove_runs_command(self):
        with patch.object(ollama, "find_executable", return_value="/usr/bin/ollama"), \
             patch.object(ollama, "run_process", return_value=(0, "ok", "")):
            assert ollama.remove_model("foo") == 0

    def test_remove_raises_on_failure(self):
        with patch.object(ollama, "find_executable", return_value="/usr/bin/ollama"), \
             patch.object(ollama, "run_process", return_value=(1, "", "bad")):
            with pytest.raises(RuntimeError):
                ollama.remove_model("foo")
