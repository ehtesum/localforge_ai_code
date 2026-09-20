"""Tests for src/ai_env/data.py."""
from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_env import data
from ai_env.data import (
    DatasetSplit,
    PreparedDataset,
    RepoChunk,
    iter_repo_chunks,
    iter_web_chunks,
    is_valid_record,
    normalize_repo_record,
    normalize_web_record,
    prepare_repo_dataset,
    prepare_web_dataset,
    read_jsonl,
    split_train_validation,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


class TestJsonlIO:
    def test_write_creates_parents(self, tmp_path):
        p = tmp_path / "deep" / "nested" / "out.jsonl"
        n = write_jsonl(p, [{"a": 1}, {"a": 2}, {"a": 3}])
        assert n == 3
        assert p.exists()
        assert p.parent.is_dir()

    def test_write_then_read_roundtrip(self, tmp_path):
        p = tmp_path / "out.jsonl"
        records = [{"x": i, "name": f"row{i}"} for i in range(10)]
        write_jsonl(p, records)
        assert list(read_jsonl(p)) == records

    def test_read_skips_blank_lines(self, tmp_path):
        p = tmp_path / "out.jsonl"
        p.write_text(
            '{"a":1}\n\n   \n{"a":2}\n',
            encoding="utf-8",
        )
        assert list(read_jsonl(p)) == [{"a": 1}, {"a": 2}]

    def test_read_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "out.jsonl"
        p.write_text(
            '{"a":1}\nnot json\n{"a":2}\n{"a":3,}\n',
            encoding="utf-8",
        )
        # Only the well-formed ones come through; the trailing-comma
        # record and the bare word are dropped.
        out = list(read_jsonl(p))
        assert out == [{"a": 1}, {"a": 2}]

    def test_write_count_zero_for_empty_iterable(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        assert write_jsonl(p, []) == 0
        # File is created but empty.
        assert p.exists()
        assert p.read_text(encoding="utf-8") == ""

    def test_write_unicode_preserved(self, tmp_path):
        p = tmp_path / "u.jsonl"
        write_jsonl(p, [{"text": "héllo wörld 你好"}])
        assert "héllo wörld 你好" in p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Record normalization
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_repo_record_shape(self):
        r = normalize_repo_record("src/main.py", "def f(): pass")
        assert r == {"text": "def f(): pass", "path": "src/main.py", "source": "repo"}

    def test_web_record_shape(self):
        r = normalize_web_record("https://example.com", "Hello")
        assert r == {"text": "Hello", "path": "https://example.com", "source": "web"}

    def test_is_valid_accepts_canonical(self):
        assert is_valid_record({"text": "x", "path": "p", "source": "repo"}) is True

    def test_is_valid_rejects_missing_keys(self):
        assert is_valid_record({"text": "x", "path": "p"}) is False
        assert is_valid_record({"path": "p", "source": "repo"}) is False
        assert is_valid_record({}) is False

    def test_is_valid_rejects_non_dict(self):
        assert is_valid_record("not a dict") is False
        assert is_valid_record(None) is False
        assert is_valid_record([1, 2]) is False


# ---------------------------------------------------------------------------
# Repo chunking
# ---------------------------------------------------------------------------


class TestIterRepoChunks:
    def test_yields_chunks(self, sample_repo):
        chunks = list(iter_repo_chunks(sample_repo))
        # sample_repo has at least 3 text files; each produces >= 1 chunk.
        assert len(chunks) >= 3
        for chunk in chunks:
            assert isinstance(chunk, RepoChunk)
            assert chunk.source == "repo"
            assert chunk.text.startswith("## Source:")

    def test_max_chars_caps_chunk_size(self, sample_repo):
        chunks = list(iter_repo_chunks(sample_repo, max_chars=50))
        # Each chunk's text is "## Source: ...\n<content>\n", so it
        # can slightly exceed max_chars due to the header overhead.
        for chunk in chunks:
            # Body of the chunk is the part after the "## Source: " line.
            body = chunk.text.split("\n", 1)[1].rstrip()
            # Allow some slack for the chunking strategy itself.
            assert len(body) <= 50 + 10

    def test_categories_filter(self, sample_repo):
        py_chunks = list(iter_repo_chunks(sample_repo, categories={"python"}))
        assert len(py_chunks) >= 1
        for c in py_chunks:
            assert c.path.endswith(".py")

    def test_all_keyword_bypasses_filter(self, sample_repo):
        all_chunks = list(iter_repo_chunks(sample_repo, categories={"all"}))
        py_chunks = list(iter_repo_chunks(sample_repo, categories={"python"}))
        assert len(all_chunks) >= len(py_chunks)


# ---------------------------------------------------------------------------
# Web chunking
# ---------------------------------------------------------------------------


class TestIterWebChunks:
    def test_yields_chunks_from_html(self):
        html = """
        <html><head><title>X</title></head>
        <body>
            <h1>Welcome</h1>
            <p>This is a paragraph with enough text to chunk well. "
            "It repeats so we can see splits.</p>
            <p>Another paragraph here with more text content for chunking.</p>
        </body></html>
        """
        chunks = list(iter_web_chunks(["https://example.com"], fetch=lambda u: html))
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.source == "web"
            assert chunk.path == "https://example.com"
            assert "Welcome" in chunk.text or "paragraph" in chunk.text

    def test_skips_url_when_fetch_returns_none(self):
        chunks = list(iter_web_chunks(
            ["https://a.example", "https://b.example"],
            fetch=lambda u: None,
        ))
        assert chunks == []

    def test_skips_url_when_fetch_raises(self):
        def bad_fetch(u):
            raise RuntimeError("nope")
        chunks = list(iter_web_chunks(["https://x"], fetch=bad_fetch))
        assert chunks == []

    def test_strips_script_tags(self):
        html = "<p>Keep me</p><script>alert('x')</script><p>And me</p>"
        chunks = list(iter_web_chunks(["u"], fetch=lambda u: html))
        text = " ".join(c.text for c in chunks)
        assert "alert" not in text
        assert "Keep me" in text
        assert "And me" in text


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------


class TestSplit:
    def _records(self, n: int) -> list[dict]:
        return [{"text": str(i), "path": f"f{i}", "source": "repo"} for i in range(n)]

    def test_basic_split(self):
        records = self._records(100)
        split = split_train_validation(records, val_split=0.2, seed=7)
        assert len(split.training) == 80
        assert len(split.validation) == 20
        assert split.total == 100

    def test_deterministic(self):
        records = self._records(50)
        a = split_train_validation(records, val_split=0.1, seed=123)
        b = split_train_validation(records, val_split=0.1, seed=123)
        assert a.training == b.training
        assert a.validation == b.validation

    def test_different_seeds_give_different_splits(self):
        records = self._records(50)
        a = split_train_validation(records, val_split=0.1, seed=1)
        b = split_train_validation(records, val_split=0.1, seed=999)
        assert a.training != b.training

    def test_no_overlap_between_train_and_val(self):
        records = self._records(100)
        split = split_train_validation(records, val_split=0.3, seed=0)
        train_ids = {r["text"] for r in split.training}
        val_ids = {r["text"] for r in split.validation}
        assert train_ids & val_ids == set()

    def test_empty_input(self):
        split = split_train_validation([], val_split=0.1, seed=0)
        assert split.training == []
        assert split.validation == []

    def test_filters_invalid_records(self):
        records = [
            {"text": "1", "path": "a", "source": "repo"},   # valid
            {"text": "2"},                                  # missing keys
            {"text": "3", "path": "b", "source": "web"},   # valid
        ]
        split = split_train_validation(records, val_split=0.5, seed=0)
        assert all(is_valid_record(r) for r in split.training)
        assert all(is_valid_record(r) for r in split.validation)
        # 2 of 3 are valid; with val_split=0.5 min_validation=1
        # we get 1 validation + 1 training.
        assert len(split.training) + len(split.validation) == 2

    def test_invalid_val_split_raises(self):
        with pytest.raises(ValueError):
            split_train_validation(self._records(10), val_split=0)
        with pytest.raises(ValueError):
            split_train_validation(self._records(10), val_split=1.0)
        with pytest.raises(ValueError):
            split_train_validation(self._records(10), val_split=-0.1)

    def test_min_validation_floor(self):
        records = self._records(5)
        split = split_train_validation(records, val_split=0.01, seed=0, min_validation=2)
        assert len(split.validation) >= 2

    def test_never_empties_training(self):
        records = self._records(3)
        split = split_train_validation(records, val_split=0.99, seed=0)
        # Even with val_split=0.99, training set is non-empty when
        # there is more than one record.
        assert len(split.training) >= 1


# ---------------------------------------------------------------------------
# High-level prepare functions
# ---------------------------------------------------------------------------


class TestPrepareRepo:
    def test_writes_train_and_validation(self, sample_repo, tmp_path):
        out = tmp_path / "data"
        result = prepare_repo_dataset(
            output_dir=out,
            root=sample_repo,
            max_chars=2000,
            val_split=0.25,
            seed=42,
        )
        assert isinstance(result, PreparedDataset)
        assert (out / "train.jsonl").exists()
        assert (out / "validation.jsonl").exists()
        assert result.training_count + result.validation_count >= 1

    def test_raises_when_repo_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(RuntimeError):
            prepare_repo_dataset(
                output_dir=tmp_path / "out",
                root=empty,
            )

    def test_categories(self, sample_repo, tmp_path):
        out = tmp_path / "out"
        result = prepare_repo_dataset(
            output_dir=out,
            root=sample_repo,
            categories={"python"},
            val_split=0.5,
            seed=0,
        )
        # All written records should come from .py files.
        for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            assert rec["path"].endswith(".py")
            assert rec["source"] == "repo"


class TestPrepareWeb:
    def test_writes_files(self, tmp_path):
        html = "<p>Some content. " * 50 + "</p>"
        out = tmp_path / "out"
        result = prepare_web_dataset(
            output_dir=out,
            urls=["https://a.example"],
            fetch=lambda u: html,
        )
        assert (out / "train.jsonl").exists()
        assert (out / "validation.jsonl").exists()
        assert result.training_count + result.validation_count >= 1

    def test_raises_when_no_chunks(self, tmp_path):
        with pytest.raises(RuntimeError):
            prepare_web_dataset(
                output_dir=tmp_path / "out",
                urls=["https://a.example"],
                fetch=lambda u: None,
            )
