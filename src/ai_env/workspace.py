"""Workspace utilities - file operations, path safety, and git helpers.

This module is the single source of truth for any code that needs to:

  * read, write, list, or search files inside a workspace root,
  * safely resolve user-supplied paths against that root,
  * ignore well-known noise directories (``.venv``, ``.git``, ``node_modules``,
    build outputs, etc.) when walking the tree,
  * perform best-effort git operations (status, diff, log).

It exists so that ``ai_env.web``, ``ai_env.train``, and the prepare_*
scripts can share one implementation instead of each re-inventing
the same ignore-list and path-safety logic.

All public functions are pure-Python, dependency-free (stdlib only),
and side-effect-free unless they take a ``root`` argument that is a
writable directory.
"""
from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


# Directories and file patterns that should be ignored when scanning
# a workspace for training data or browsing. Keep the list small and
# obvious; large/auto-generated trees slow every consumer.
IGNORE_PATH_PARTS: frozenset[str] = frozenset({
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    ".env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".DS_Store",
    "checkpoints",
    "ollama_data",
    "roocode_config",
    "training_data",
})

IGNORE_FILE_PATTERNS: tuple[str, ...] = (
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.egg-info",
    "*.log",
    "*.pid",
    "*.bin",
    "*.safetensors",
    "*.gguf",
    "*.onnx",
)

# Common source-code file extensions. Keep this explicit (no wildcards)
# so that we never accidentally slurp binary blobs into training data.
TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".vue", ".svelte",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".md", ".rst", ".txt", ".adoc",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".sql",
    ".dockerfile", ".containerfile",
    ".xml", ".proto", ".graphql", ".gql",
    ".java", ".kt", ".kts", ".scala", ".groovy",
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx",
    ".cs", ".fs", ".vb",
    ".go", ".rs", ".rb", ".php", ".pl", ".lua", ".r",
    ".swift", ".m", ".mm",
    ".ex", ".exs", ".erl", ".hrl", ".clj", ".cljs", ".cljc",
    ".dart", ".jl", ".zig",
})

# Filenames that are always treated as text even without an extension.
TEXT_FILENAMES: frozenset[str] = frozenset({
    "dockerfile", "containerfile", "makefile", "rakefile",
    "gemfile", "podfile", "procfile", "readme", "license",
    "todo", "changelog", "authors", "contributors",
})


@dataclass(frozen=True)
class TextFile:
    """A text file discovered inside a workspace."""

    path: Path            # absolute, resolved path
    relative: str         # path relative to the workspace root, POSIX form
    text: str             # file contents (utf-8, errors='replace')


def should_ignore(path: Path) -> bool:
    """Return True if any path component or filename matches an ignore rule.

    Matching is case-insensitive and uses the directory-name set
    ``IGNORE_PATH_PARTS`` plus the glob patterns in ``IGNORE_FILE_PATTERNS``.
    """
    lowered_parts = [part.lower() for part in path.parts]
    for part in lowered_parts:
        if part in IGNORE_PATH_PARTS:
            return True
    name_lower = path.name.lower()
    for pattern in IGNORE_FILE_PATTERNS:
        if fnmatch.fnmatch(name_lower, pattern):
            return True
    return False


def is_text_file(path: Path) -> bool:
    """Heuristic: is ``path`` likely a text file we want to read?"""
    if path.name.lower() in TEXT_FILENAMES:
        return True
    return path.suffix.lower() in TEXT_EXTENSIONS


def safe_workspace_path(root: Path, requested_path: str) -> Path:
    """Resolve ``requested_path`` against ``root`` and verify it stays inside.

    Raises ``PermissionError`` if the resolved path escapes the workspace.
    Always returns an absolute, resolved ``Path``.
    """
    normalized = str(requested_path or "").lstrip("/\\")
    target = (root / normalized).resolve()
    root_resolved = root.resolve()
    if root_resolved != target and root_resolved not in target.parents:
        raise PermissionError("Path is outside the allowed workspace root")
    return target


def read_text_file(path: Path, max_bytes: int = 2_000_000) -> str | None:
    """Read a text file safely.

    Returns ``None`` for files that are too large, unreadable, or that
    raise any decoding error. Binary files are detected by attempting
    a partial read and looking for NUL bytes in the first 8 KB.
    """
    try:
        if not path.exists() or not path.is_file():
            return None
        size = path.stat().st_size
        if size > max_bytes:
            return None
        with path.open("rb") as fh:
            raw = fh.read()
        # Sniff for binary content; tolerate small NULs that may appear
        # in source code (rare but possible in string literals).
        if b"\x00" in raw[:8192]:
            return None
        return raw.decode("utf-8", errors="replace")
    except OSError:
        return None


def list_text_files(
    root: Path,
    *,
    max_files: int | None = None,
    follow_symlinks: bool = False,
) -> list[TextFile]:
    """Walk ``root`` and return a list of text files, respecting ignore rules.

    The result is sorted by relative path for stable output.
    """
    if not root.exists():
        return []
    found: list[TextFile] = []
    for item in root.rglob("*"):
        if not item.is_file(follow_symlinks=follow_symlinks):
            continue
        if should_ignore(item):
            continue
        if not is_text_file(item):
            continue
        text = read_text_file(item)
        if text is None:
            continue
        rel = item.relative_to(root).as_posix()
        found.append(TextFile(path=item, relative=rel, text=text))
        if max_files is not None and len(found) >= max_files:
            break
    found.sort(key=lambda tf: tf.relative)
    return found


def write_workspace_file(root: Path, requested_path: str, content: str) -> Path:
    """Write ``content`` to a file inside ``root``, creating parents.

    Returns the resolved absolute path of the written file.
    """
    target = safe_workspace_path(root, requested_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def iter_text_files(root: Path) -> Iterator[TextFile]:
    """Generator variant of :func:`list_text_files` for large workspaces."""
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        if should_ignore(item):
            continue
        if not is_text_file(item):
            continue
        text = read_text_file(item)
        if text is None:
            continue
        rel = item.relative_to(root).as_posix()
        yield TextFile(path=item, relative=rel, text=text)


# ---------------------------------------------------------------------------
# Git operations (best-effort; no error if git is missing or not a repo).
# ---------------------------------------------------------------------------


def _run_git(root: Path, args: Iterable[str]) -> str:
    """Run a git command inside ``root`` and return its stdout (stripped)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return ""
    return (result.stdout or "").strip()


def is_git_repo(root: Path) -> bool:
    """True if ``root`` is inside a working git repository."""
    if not shutil.which("git"):
        return False
    code = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True, check=False,
    ).returncode
    return code == 0


def git_status(root: Path) -> str:
    """Return ``git status --short`` output, or an empty string on failure."""
    return _run_git(root, ["status", "--short"])


def git_diff(root: Path, path: str | None = None) -> str:
    """Return ``git diff`` (optionally for a single path), or empty on failure."""
    if path:
        return _run_git(root, ["diff", "--", path])
    return _run_git(root, ["diff"])


def git_diff_stat(root: Path) -> str:
    """Return ``git diff --stat`` output, or empty on failure."""
    return _run_git(root, ["diff", "--stat"])


@dataclass(frozen=True)
class CommitPreview:
    """A short summary of pending changes, suitable for a chat message."""

    summary: str
    status: str
    diff_stat: str

    def to_dict(self) -> dict:
        return {"summary": self.summary, "status": self.status, "diff_stat": self.diff_stat}


def preview_commit(root: Path) -> CommitPreview:
    """Inspect the working tree and return a short human-readable preview."""
    status = git_status(root)
    diff_stat = git_diff_stat(root)
    if not status:
        return CommitPreview(
            summary="No tracked changes in the workspace.",
            status=status,
            diff_stat=diff_stat,
        )
    changed = [line for line in status.splitlines() if line.strip()]
    if len(changed) == 1:
        # Format: " M path/to/file" - filename is the third field.
        filename = changed[0].split(maxsplit=1)[-1]
        summary = f"Update {filename}"
    else:
        summary = f"Update {len(changed)} file(s)"
    return CommitPreview(
        summary=summary,
        status=status,
        diff_stat=diff_stat or "No diffstat available.",
    )


def disk_usage(path: Path) -> dict:
    """Return ``{total, used, free}`` disk usage in bytes for ``path``."""
    usage = shutil.disk_usage(path)
    return {"total": usage.total, "used": usage.used, "free": usage.free}


def system_snapshot() -> dict:
    """A small JSON-serializable snapshot of the host system.

    Useful for ``/api/status`` endpoints. ``psutil`` is optional - if
    not installed, memory info is omitted.
    """
    info: dict = {
        "cpu_count": os.cpu_count() or 1,
        "python_version": _python_version_string(),
    }
    try:
        import psutil  # type: ignore
        mem = psutil.virtual_memory()
        info.update({
            "memory_total": mem.total,
            "memory_available": mem.available,
            "memory_percent": mem.percent,
        })
    except ImportError:
        pass
    return info


def _python_version_string() -> str:
    import sys
    return sys.version.split()[0]


# ---------------------------------------------------------------------------
# Chunking helpers (shared between train.py and the Web UI data layer).
# ---------------------------------------------------------------------------


_SENTENCE_END = re.compile(r"(?<=[.!?\n])\s+")


def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` characters.

    Tries to break on sentence/paragraph boundaries; falls back to a
    hard split if a single sentence exceeds ``max_chars``.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    # First, split on natural boundaries.
    pieces = _SENTENCE_END.split(text)
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if len(current) + len(piece) + 1 <= max_chars:
            current = f"{current} {piece}".strip() if current else piece
            continue
        if current:
            chunks.append(current)
        if len(piece) > max_chars:
            # Hard-split oversized piece.
            for i in range(0, len(piece), max_chars):
                chunks.append(piece[i:i + max_chars])
            current = ""
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


__all__ = [
    "IGNORE_PATH_PARTS",
    "IGNORE_FILE_PATTERNS",
    "TEXT_EXTENSIONS",
    "TEXT_FILENAMES",
    "TextFile",
    "should_ignore",
    "is_text_file",
    "safe_workspace_path",
    "read_text_file",
    "list_text_files",
    "write_workspace_file",
    "iter_text_files",
    "is_git_repo",
    "git_status",
    "git_diff",
    "git_diff_stat",
    "CommitPreview",
    "preview_commit",
    "disk_usage",
    "system_snapshot",
    "chunk_text",
]
