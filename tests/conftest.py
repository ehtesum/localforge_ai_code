"""Shared pytest fixtures and configuration for the ai_env test suite."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Make ``src/`` importable as ``ai_env`` regardless of where pytest is invoked from.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    """A clean, writable temporary directory for workspace tests."""
    return tmp_path


@pytest.fixture
def sample_repo(workspace_dir: Path) -> Path:
    """Create a tiny fake repository with a mix of text and binary files."""
    (workspace_dir / "src").mkdir()
    (workspace_dir / "src" / "app.py").write_text(
        "def add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    (workspace_dir / "src" / "README.md").write_text(
        "# Sample\n\nA tiny project.\n",
        encoding="utf-8",
    )
    (workspace_dir / "config.yaml").write_text(
        "name: sample\nversion: 0.1.0\n",
        encoding="utf-8",
    )
    (workspace_dir / "data").mkdir()
    (workspace_dir / "data" / "blob.bin").write_bytes(b"\x00\x01\x02\x03not text")
    (workspace_dir / ".venv").mkdir()
    (workspace_dir / ".venv" / "lib.py").write_text("# ignored\n", encoding="utf-8")
    return workspace_dir
