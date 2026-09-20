"""Tests for src/ai_env/workspace.py."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_env import workspace
from ai_env.workspace import (
    CommitPreview,
    IGNORE_PATH_PARTS,
    TextFile,
    chunk_text,
    disk_usage,
    git_diff,
    git_diff_stat,
    git_status,
    is_git_repo,
    is_text_file,
    iter_text_files,
    list_text_files,
    preview_commit,
    read_text_file,
    safe_workspace_path,
    should_ignore,
    system_snapshot,
    write_workspace_file,
)


# ---------------------------------------------------------------------------
# should_ignore / is_text_file
# ---------------------------------------------------------------------------


class TestShouldIgnore:
    def test_ignores_venv(self):
        assert should_ignore(Path(".venv/lib/site-packages/foo.py")) is True

    def test_ignores_pycache(self):
        assert should_ignore(Path("src/__pycache__/x.cpython-313.pyc")) is True

    def test_ignores_git(self):
        assert should_ignore(Path(".git/HEAD")) is True

    def test_ignores_pyc_files(self):
        assert should_ignore(Path("src/foo.pyc")) is True

    def test_ignores_safetensors(self):
        assert should_ignore(Path("checkpoints/model.safetensors")) is True

    def test_does_not_ignore_source_code(self):
        assert should_ignore(Path("src/main.py")) is False
        assert should_ignore(Path("README.md")) is False

    def test_case_insensitive(self):
        assert should_ignore(Path(".VENV/x.py")) is True
        assert should_ignore(Path("Foo.PY")) is False  # pattern is *.pyc, not *.PY


class TestIsTextFile:
    @pytest.mark.parametrize("name", [
        "Dockerfile", "Makefile", "README", "LICENSE",
        "CHANGELOG", "Procfile", "containerfile",
    ])
    def test_known_text_filenames(self, name):
        assert is_text_file(Path(name)) is True

    @pytest.mark.parametrize("suffix", [
        ".py", ".js", ".ts", ".tsx", ".vue", ".go", ".rs",
        ".md", ".json", ".yaml", ".toml", ".sh", ".ps1",
    ])
    def test_known_text_extensions(self, suffix):
        assert is_text_file(Path(f"file{suffix}")) is True

    @pytest.mark.parametrize("suffix", [
        ".png", ".jpg", ".zip", ".tar", ".exe", ".so",
        ".pyc", ".bin", ".gguf", ".safetensors",
    ])
    def test_binary_extensions(self, suffix):
        assert is_text_file(Path(f"file{suffix}")) is False


# ---------------------------------------------------------------------------
# safe_workspace_path
# ---------------------------------------------------------------------------


class TestSafeWorkspacePath:
    def test_resolves_inside(self, tmp_path):
        target = safe_workspace_path(tmp_path, "src/main.py")
        assert target == (tmp_path / "src" / "main.py").resolve()

    def test_strips_leading_slash(self, tmp_path):
        a = safe_workspace_path(tmp_path, "/src/main.py")
        b = safe_workspace_path(tmp_path, "src/main.py")
        assert a == b

    def test_strips_leading_backslash(self, tmp_path):
        a = safe_workspace_path(tmp_path, "\\src\\main.py")
        b = safe_workspace_path(tmp_path, "src/main.py")
        assert a == b

    def test_rejects_escape(self, tmp_path):
        with pytest.raises(PermissionError):
            safe_workspace_path(tmp_path, "../../../etc/passwd")

    def test_rejects_absolute_outside(self, tmp_path):
        with pytest.raises(PermissionError):
            safe_workspace_path(tmp_path, "C:/Windows/System32/notepad.exe")

    def test_handles_empty_string(self, tmp_path):
        target = safe_workspace_path(tmp_path, "")
        assert target == tmp_path.resolve()


# ---------------------------------------------------------------------------
# read_text_file / write_workspace_file
# ---------------------------------------------------------------------------


class TestReadTextFile:
    def test_reads_utf8(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("hello", encoding="utf-8")
        assert read_text_file(f) == "hello"

    def test_returns_none_for_missing(self, tmp_path):
        assert read_text_file(tmp_path / "nope.txt") is None

    def test_returns_none_for_binary(self, tmp_path):
        f = tmp_path / "blob.bin"
        f.write_bytes(b"\x00\x01\x02binary\x00")
        assert read_text_file(f) is None

    def test_returns_none_for_oversize(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 10)
        assert read_text_file(f, max_bytes=5) is None

    def test_replaces_decode_errors(self, tmp_path):
        f = tmp_path / "mixed.txt"
        f.write_bytes(b"hello\xc3\x28world")  # invalid utf-8 sequence
        result = read_text_file(f)
        assert result is not None
        assert "hello" in result


class TestWriteWorkspaceFile:
    def test_writes_and_creates_parents(self, tmp_path):
        out = write_workspace_file(tmp_path, "deep/nested/file.txt", "hi")
        assert out.exists()
        assert out.read_text(encoding="utf-8") == "hi"

    def test_rejects_escape(self, tmp_path):
        with pytest.raises(PermissionError):
            write_workspace_file(tmp_path, "../escape.txt", "x")


# ---------------------------------------------------------------------------
# list_text_files / iter_text_files
# ---------------------------------------------------------------------------


class TestListTextFiles:
    def test_finds_text_files(self, sample_repo):
        files = list_text_files(sample_repo)
        rels = {f.relative for f in files}
        assert "src/app.py" in rels
        # sample_repo puts the readme under src/ (see tests/conftest.py)
        assert any(r.endswith("README.md") for r in rels)

    def test_skips_ignored_dirs(self, sample_repo):
        files = list_text_files(sample_repo)
        rels = {f.relative for f in files}
        assert ".venv/lib.py" not in rels
        assert "data/blob.bin" not in rels

    def test_sorted_by_relative(self, sample_repo):
        files = list_text_files(sample_repo)
        rels = [f.relative for f in files]
        assert rels == sorted(rels)

    def test_max_files(self, sample_repo):
        files = list_text_files(sample_repo, max_files=1)
        assert len(files) == 1

    def test_returns_text_file_instances(self, sample_repo):
        files = list_text_files(sample_repo)
        for f in files:
            assert isinstance(f, TextFile)
            assert isinstance(f.text, str)

    def test_empty_root(self, tmp_path):
        assert list_text_files(tmp_path) == []

    def test_missing_root(self, tmp_path):
        assert list_text_files(tmp_path / "does-not-exist") == []


class TestIterTextFiles:
    def test_yields_same_as_list(self, sample_repo):
        a = list(iter_text_files(sample_repo))
        b = list_text_files(sample_repo)
        assert {f.relative for f in a} == {f.relative for f in b}


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


class TestIsGitRepo:
    def test_false_when_no_git(self, monkeypatch, tmp_path):
        monkeypatch.setattr(workspace.shutil, "which", lambda x: None)
        assert is_git_repo(tmp_path) is False

    def test_false_when_not_a_repo(self, monkeypatch, tmp_path):
        monkeypatch.setattr(workspace.shutil, "which", lambda x: "/usr/bin/git")
        with patch.object(subprocess, "run") as run:
            run.return_value.returncode = 128
            assert is_git_repo(tmp_path) is False

    def test_true_when_in_repo(self, monkeypatch, tmp_path):
        monkeypatch.setattr(workspace.shutil, "which", lambda x: "/usr/bin/git")
        with patch.object(subprocess, "run") as run:
            run.return_value.returncode = 0
            assert is_git_repo(tmp_path) is True


class TestGitCommands:
    def test_git_status_returns_stdout(self, monkeypatch):
        with patch.object(subprocess, "run") as run:
            run.return_value.stdout = "M foo.py\n"
            run.return_value.returncode = 0
            assert git_status(Path(".")) == "M foo.py"

    def test_git_status_returns_empty_on_error(self, monkeypatch):
        with patch.object(subprocess, "run") as run:
            run.return_value.stdout = ""
            run.return_value.returncode = 128
            assert git_status(Path(".")) == ""

    def test_git_status_returns_empty_when_git_missing(self, monkeypatch):
        def boom(*args, **kwargs):
            raise FileNotFoundError("git not found")
        with patch.object(subprocess, "run", side_effect=boom):
            assert git_status(Path(".")) == ""

    def test_git_diff_with_path(self, monkeypatch):
        with patch.object(subprocess, "run") as run:
            run.return_value.stdout = "diff --git a/x b/x\n"
            run.return_value.returncode = 0
            out = git_diff(Path("."), path="x")
            assert "diff --git" in out
            assert run.call_args.args[0] == ["git", "-C", ".", "diff", "--", "x"]

    def test_git_diff_stat(self, monkeypatch):
        with patch.object(subprocess, "run") as run:
            run.return_value.stdout = " x | 1 +\n"
            run.return_value.returncode = 0
            assert "x | 1" in git_diff_stat(Path("."))


class TestPreviewCommit:
    def test_no_changes(self, monkeypatch):
        monkeypatch.setattr(workspace, "git_status", lambda r: "")
        monkeypatch.setattr(workspace, "git_diff_stat", lambda r: "")
        preview = preview_commit(Path("."))
        assert isinstance(preview, CommitPreview)
        assert "No tracked changes" in preview.summary

    def test_single_change(self, monkeypatch):
        monkeypatch.setattr(workspace, "git_status", lambda r: " M foo.py")
        monkeypatch.setattr(workspace, "git_diff_stat", lambda r: " foo.py | 2 +-")
        preview = preview_commit(Path("."))
        assert preview.summary == "Update foo.py"

    def test_multiple_changes(self, monkeypatch):
        monkeypatch.setattr(
            workspace, "git_status",
            lambda r: " M a.py\n M b.py\n M c.py",
        )
        monkeypatch.setattr(workspace, "git_diff_stat", lambda r: "")
        preview = preview_commit(Path("."))
        assert preview.summary == "Update 3 file(s)"

    def test_to_dict(self):
        p = CommitPreview(summary="x", status="", diff_stat="")
        assert p.to_dict() == {"summary": "x", "status": "", "diff_stat": ""}


# ---------------------------------------------------------------------------
# System / disk helpers
# ---------------------------------------------------------------------------


class TestDiskUsage:
    def test_returns_dict_with_three_keys(self, tmp_path):
        u = disk_usage(tmp_path)
        assert set(u.keys()) == {"total", "used", "free"}
        assert all(isinstance(v, int) for v in u.values())


class TestSystemSnapshot:
    def test_contains_cpu_and_python(self):
        s = system_snapshot()
        assert "cpu_count" in s
        assert "python_version" in s
        assert isinstance(s["cpu_count"], int)


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_short_text_yields_one_chunk(self):
        assert chunk_text("hi", max_chars=10) == ["hi"]

    def test_empty_text_yields_nothing(self):
        assert chunk_text("", max_chars=10) == []
        assert chunk_text("   \n  ", max_chars=10) == []

    def test_splits_on_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, max_chars=30)
        assert len(chunks) >= 2
        assert all(len(c) <= 30 for c in chunks)

    def test_preserves_full_text(self):
        text = "A" * 100 + ". " + "B" * 100 + ". " + "C" * 100
        chunks = chunk_text(text, max_chars=120)
        rejoined = " ".join(chunks)
        # Some whitespace normalization happens at chunk boundaries; just
        # verify that no content is lost.
        assert "A" * 100 in rejoined
        assert "B" * 100 in rejoined
        assert "C" * 100 in rejoined

    def test_hard_splits_oversized_piece(self):
        text = "x" * 5000
        chunks = chunk_text(text, max_chars=1000)
        assert all(len(c) <= 1000 for c in chunks)
        assert "".join(chunks) == text
