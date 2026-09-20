"""Tests for src/ai_env/utils.py."""
from __future__ import annotations

import json

import pytest

from ai_env.utils import find_executable, run_process, write_json


class TestFindExecutable:
    def test_returns_path_for_existing_command(self):
        # ``python`` is always on PATH in the test environment
        path = find_executable("python")
        assert path is not None
        assert "python" in path.lower()

    def test_returns_none_for_missing_command(self):
        assert find_executable("definitely-not-a-real-binary-xyz123") is None


class TestRunProcess:
    def test_captures_stdout(self):
        code, out, err = run_process(
            ["python", "-c", "print('hello')"], capture_output=True
        )
        assert code == 0
        assert out == "hello"
        assert err == ""

    def test_captures_stderr(self):
        code, out, err = run_process(
            ["python", "-c", "import sys; sys.stderr.write('oops')"],
            capture_output=True,
        )
        assert code == 0
        assert err == "oops"

    def test_propagates_returncode(self):
        code, _, _ = run_process(
            ["python", "-c", "raise SystemExit(7)"], capture_output=True
        )
        assert code == 7

    def test_handles_missing_command(self):
        with pytest.raises(FileNotFoundError):
            run_process(["__nope__"], capture_output=True)


class TestWriteJson:
    def test_writes_valid_json(self, tmp_path):
        target = tmp_path / "out.json"
        write_json(target, {"a": 1, "b": [1, 2, 3]})
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": [1, 2, 3]}

    def test_creates_missing_parents(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "out.json"
        write_json(target, {"x": True})
        assert target.exists()

    def test_uses_pretty_indent(self, tmp_path):
        target = tmp_path / "out.json"
        write_json(target, {"k": "v"})
        text = target.read_text(encoding="utf-8")
        # indent=2 -> newlines present
        assert "\n" in text
