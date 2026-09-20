"""Tests for src/ai_env/model_loader.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_env.model_loader import detect_device, resolve_checkpoint


class TestDetectDevice:
    def test_returns_cpu_when_cuda_unavailable(self, monkeypatch):
        # Force torch.cuda.is_available() -> False
        import ai_env.model_loader as ml

        class _FakeCuda:
            @staticmethod
            def is_available():
                return False

        class _FakeTorch:
            cuda = _FakeCuda()

        monkeypatch.setattr(ml, "torch", _FakeTorch)
        assert detect_device() == "cpu"

    def test_returns_cpu_when_prefer_cuda_false(self):
        assert detect_device(prefer_cuda=False) == "cpu"


class TestResolveCheckpoint:
    def test_returns_none_for_missing_dir(self, tmp_path):
        path, name = resolve_checkpoint(tmp_path / "missing")
        assert path is None
        assert name is None

    def test_picks_latest_checkpoint_subdir(self, tmp_path):
        for n in (1, 2, 3):
            (tmp_path / f"checkpoint-{n * 100}").mkdir()
        path, name = resolve_checkpoint(tmp_path)
        assert name == "checkpoint-300"
        assert path == tmp_path / "checkpoint-300"

    def test_uses_nonempty_dir_when_no_checkpoints(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        path, name = resolve_checkpoint(tmp_path)
        assert path == tmp_path
        assert name is None

    def test_returns_none_for_empty_dir(self, tmp_path):
        path, name = resolve_checkpoint(tmp_path)
        assert path is None
        assert name is None

    def test_falls_back_to_remote_when_no_local(self, tmp_path):
        path, name = resolve_checkpoint(tmp_path, fallback_remote="hf/model")
        assert path == Path("hf/model")
        assert name is None
