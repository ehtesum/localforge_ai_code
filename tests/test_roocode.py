"""Tests for src/ai_env/roocode.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_env import roocode
from ai_env.config import ROO_CONFIG_PATH


class TestEnsureRooCodeConfig:
    def test_creates_file_when_missing(self, tmp_path, monkeypatch):
        target = tmp_path / "roo.json"
        monkeypatch.setattr(roocode, "ROO_CONFIG_PATH", target)
        result = roocode.ensure_roocode_config()
        assert result == target
        assert target.exists()
        data = json.loads(target.read_text(encoding="utf-8"))
        assert "llm" in data
        assert "editor" in data
        assert "backend" in data

    def test_is_idempotent(self, tmp_path, monkeypatch):
        target = tmp_path / "roo.json"
        monkeypatch.setattr(roocode, "ROO_CONFIG_PATH", target)
        roocode.ensure_roocode_config()
        first = target.read_text(encoding="utf-8")
        roocode.ensure_roocode_config()
        second = target.read_text(encoding="utf-8")
        assert first == second  # not overwritten

    def test_contains_expected_models(self, tmp_path, monkeypatch):
        target = tmp_path / "roo.json"
        monkeypatch.setattr(roocode, "ROO_CONFIG_PATH", target)
        roocode.ensure_roocode_config()
        data = json.loads(target.read_text(encoding="utf-8"))
        models = data["llm"]["models"]
        assert "qwen2.5-coder:7b" in models
        assert "qwen3:8b" in models


class TestReadRooCodeConfig:
    def test_reads_existing(self, tmp_path, monkeypatch):
        target = tmp_path / "roo.json"
        target.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
        monkeypatch.setattr(roocode, "ROO_CONFIG_PATH", target)
        assert roocode.read_roocode_config() == {"hello": "world"}

    def test_raises_when_missing(self, tmp_path, monkeypatch):
        target = tmp_path / "missing.json"
        monkeypatch.setattr(roocode, "ROO_CONFIG_PATH", target)
        with pytest.raises(FileNotFoundError):
            roocode.read_roocode_config()


class TestDefaultRooCodeConfig:
    def test_default_has_endpoint(self):
        assert "endpoint" in roocode.DEFAULT_ROO_CODE_CONFIG["llm"]
        assert roocode.DEFAULT_ROO_CODE_CONFIG["llm"]["provider"] == "ollama"
