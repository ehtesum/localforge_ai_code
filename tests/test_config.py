"""Tests for src/ai_env/config.py."""
from __future__ import annotations

from pathlib import Path

from ai_env import config


class TestConfigConstants:
    def test_root_is_absolute(self):
        assert config.ROOT.is_absolute()

    def test_root_points_to_repo_root(self):
        # The src/ai_env/config.py file is at src/ai_env/, so parents[2] is the repo root.
        assert config.ROOT == Path(config.__file__).resolve().parents[2]

    def test_default_model_in_defaults(self):
        assert config.DEFAULT_MODEL in config.DEFAULT_MODELS

    def test_ports_are_integers(self):
        assert isinstance(config.OLLAMA_PORT, int)
        assert isinstance(config.ROOCODE_PORT, int)
        assert 1 <= config.OLLAMA_PORT <= 65535
        assert 1 <= config.ROOCODE_PORT <= 65535

    def test_paths_under_repo(self):
        for p in (config.ROO_CONFIG_PATH, config.DOCKER_COMPOSE_PATH, config.OLLAMA_DATA_PATH):
            assert p.is_absolute()
            assert str(p).startswith(str(config.ROOT))
