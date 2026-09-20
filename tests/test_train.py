"""Tests for src/ai_env/train.py (data preparation)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_env import train


class TestIsTextFile:
    @pytest.mark.parametrize("name,suffix,expected", [
        ("app.py", ".py", True),
        ("README.md", ".md", True),
        ("script.sh", ".sh", True),
        ("Dockerfile", "", True),
        ("Makefile", "", True),
        ("image.png", ".png", False),
        ("data.bin", ".bin", False),
    ])
    def test_detection(self, name, suffix, expected, tmp_path):
        # ``tmp_path`` not used directly, just to give the test a real cwd
        p = tmp_path / name
        p.write_text("hello", encoding="utf-8")
        assert train.is_text_file(p) is expected


class TestShouldIgnore:
    def test_ignores_venv(self, tmp_path):
        p = tmp_path / ".venv" / "lib" / "site.py"
        assert train.should_ignore(p) is True

    def test_ignores_git(self, tmp_path):
        p = tmp_path / ".git" / "config"
        assert train.should_ignore(p) is True

    def test_ignores_pycache(self, tmp_path):
        p = tmp_path / "src" / "__pycache__" / "x.cpython-311.pyc"
        assert train.should_ignore(p) is True

    def test_does_not_ignore_normal(self, tmp_path):
        p = tmp_path / "src" / "app.py"
        assert train.should_ignore(p) is False


class TestMatchesCategory:
    def test_all_matches_everything(self, tmp_path):
        p = tmp_path / "foo.xyz"
        assert train.matches_category(p, {"all"}) is True

    def test_python_category(self, tmp_path):
        p = tmp_path / "app.py"
        assert train.matches_category(p, {"python"}) is True

    def test_frontend_category(self, tmp_path):
        p = tmp_path / "component.tsx"
        assert train.matches_category(p, {"frontend"}) is True

    def test_unknown_category(self, tmp_path):
        p = tmp_path / "app.py"
        assert train.matches_category(p, {"backend"}) is False

    def test_dockerfile_matches_devops(self, tmp_path):
        p = tmp_path / "Dockerfile"
        assert train.matches_category(p, {"devops"}) is True


class TestChunkText:
    def test_short_text_yields_one_chunk(self):
        chunks = list(train.chunk_text("hello world", max_chars=100))
        assert chunks == ["hello world"]

    def test_long_text_is_split(self):
        text = "a" * 2500
        chunks = list(train.chunk_text(text, max_chars=1000))
        assert all(len(c) <= 1000 for c in chunks)
        assert "".join(chunks) == text

    def test_splits_on_line_boundaries(self):
        text = "\n".join(["x" * 50] * 30)  # 30 lines of 50 chars
        chunks = list(train.chunk_text(text, max_chars=200))
        # Each chunk should be <= 200 chars
        assert all(len(c) <= 200 for c in chunks)
        # And the content is preserved
        assert "".join(chunks).replace("\n", "") == "x" * 1500


class TestReadTextFile:
    def test_reads_utf8(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("hello", encoding="utf-8")
        assert train.read_text_file(p) == "hello"

    def test_handles_unreadable(self, tmp_path, monkeypatch):
        p = tmp_path / "f.txt"
        p.write_text("x", encoding="utf-8")
        # Force read_text to raise to exercise the fallback path
        from ai_env import train as t

        def _raise(*args, **kwargs):
            raise OSError("boom")

        monkeypatch.setattr(Path, "open", _raise)
        assert t.read_text_file(p) is None


class TestWriteJsonl:
    def test_writes_records_as_lines(self, tmp_path):
        target = tmp_path / "out.jsonl"
        records = [{"a": 1}, {"b": 2}, {"c": "你好"}]
        train.write_jsonl(target, records)
        lines = target.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[2]) == {"c": "你好"}


class TestIterRepoChunks:
    def test_skips_ignored_dirs(self, sample_repo):
        chunks = list(train.iter_repo_chunks(sample_repo, categories={"all"}))
        paths = {c["path"] for c in chunks}
        assert "src/app.py" in paths
        assert "src/README.md" in paths
        assert "config.yaml" in paths
        # ``.venv`` contents are ignored
        assert not any(".venv" in p for p in paths)
        # Binary file is not a text file
        assert not any("blob.bin" in p for p in paths)

    def test_chunk_has_source_header(self, sample_repo):
        chunks = list(train.iter_repo_chunks(sample_repo, categories={"all"}))
        assert any(c["text"].startswith("## Source: src/app.py") for c in chunks)

    def test_category_filter(self, sample_repo):
        chunks = list(train.iter_repo_chunks(sample_repo, categories={"python"}))
        paths = {c["path"] for c in chunks}
        assert "src/app.py" in paths
        assert "config.yaml" not in paths


class TestPrepareData:
    def test_creates_train_and_validation_files(self, sample_repo, tmp_path, monkeypatch):
        # Override ROOT-like behaviour by passing an explicit output dir
        chunks = list(train.iter_repo_chunks(sample_repo, categories={"all"}))
        assert chunks  # sanity

        # Write directly using prepare_data on a temp repo
        out = tmp_path / "training_data"
        # Patch the module-level ROOT to our sample so the walker finds files
        monkeypatch.setattr(train, "ROOT", sample_repo)
        train.prepare_data(output_dir=out, max_chars=2000, val_split=0.5, seed=0)

        train_file = out / "train.jsonl"
        val_file = out / "validation.jsonl"
        assert train_file.exists()
        assert val_file.exists()

        train_lines = [json.loads(l) for l in train_file.read_text(encoding="utf-8").splitlines()]
        val_lines = [json.loads(l) for l in val_file.read_text(encoding="utf-8").splitlines()]
        assert train_lines and val_lines
        # With val_split=0.5 and a small dataset, both partitions must be non-empty
        assert all("text" in r and "path" in r for r in train_lines + val_lines)

    def test_raises_when_no_chunks(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "node_modules").mkdir()
        (empty / "node_modules" / "x.js").write_text("x", encoding="utf-8")
        monkeypatch.setattr(train, "ROOT", empty)
        with pytest.raises(RuntimeError):
            train.prepare_data(output_dir=tmp_path / "out", categories={"python"})
