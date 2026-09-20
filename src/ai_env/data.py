"""Dataset preparation - JSONL read/write, train/validation split, normalization.

This module is the single source of truth for transforming raw records
(coming from the local repo, web snapshots, or future sources) into the
JSONL format consumed by the fine-tuning pipeline.

  * :func:`read_jsonl` / :func:`write_jsonl` - streaming I/O for JSONL
  * :func:`iter_repo_chunks` - walk a repo via ``ai_env.workspace`` and
    yield per-chunk records
  * :func:`iter_web_chunks` - fetch URLs and yield per-chunk records
    (moved from ``ai_env.web`` to break the inverted dependency)
  * :func:`split_train_validation` - deterministic train/val split
  * :func:`normalize_repo_record` / :func:`normalize_web_record` -
    canonical record shape: ``{text, path, source}``
  * :func:`prepare_repo_dataset` / :func:`prepare_web_dataset` -
    high-level "give me a train.jsonl + validation.jsonl" entry points

The module has no third-party dependencies (stdlib only) and is safe
to import from both the Web UI and the training scripts.
"""
from __future__ import annotations

import html as html_lib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from .workspace import (
    IGNORE_PATH_PARTS,
    TextFile,
    chunk_text,
    iter_text_files,
    list_text_files,
    should_ignore,
)


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield records from a JSONL file, skipping blank lines.

    Yields ``dict`` instances. Malformed lines are silently skipped
    (we never want a single bad line to abort a multi-thousand-record
    training run).
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    """Write ``records`` to ``path`` as JSONL. Returns the count written.

    Parent directories are created automatically. Records that are not
    JSON-serializable raise ``TypeError``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Record normalization
# ---------------------------------------------------------------------------


REQUIRED_KEYS = frozenset({"text", "path", "source"})


def is_valid_record(record: dict) -> bool:
    """True if ``record`` has the canonical shape produced by this module."""
    if not isinstance(record, dict):
        return False
    return REQUIRED_KEYS.issubset(record.keys())


def normalize_repo_record(relative_path: str, text: str) -> dict:
    """Build a canonical record for a single repo chunk."""
    return {
        "text": text,
        "path": relative_path,
        "source": "repo",
    }


def normalize_web_record(url: str, text: str) -> dict:
    """Build a canonical record for a single web chunk."""
    return {
        "text": text,
        "path": url,
        "source": "web",
    }


# ---------------------------------------------------------------------------
# Repo chunking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoChunk:
    """A single training chunk produced from a repo file."""

    text: str
    path: str
    source: str = "repo"

    def to_dict(self) -> dict:
        return {"text": self.text, "path": self.path, "source": self.source}


def iter_repo_chunks(
    root: Path,
    max_chars: int = 2000,
    categories: set[str] | None = None,
) -> Iterator[RepoChunk]:
    """Walk ``root`` and yield a ``RepoChunk`` per text chunk per file.

    Args:
        root: workspace root to walk
        max_chars: maximum characters per chunk (see
            :func:`ai_env.workspace.chunk_text`)
        categories: optional set of category filters. Currently
            supported: ``"all"`` (default; everything), ``"python"``
            (``.py`` and ``.pyi`` only), ``"devops"`` (Dockerfile,
            yaml, toml, sh, ps1, md), ``"frontend"`` (html/css/js/ts/
            vue/svelte). Use ``"all"`` explicitly to bypass filtering.
    """
    categories = categories or {"all"}
    for text_file in iter_text_files(Path(root)):
        if not _matches_category(text_file.relative, categories):
            continue
        for chunk in chunk_text(text_file.text, max_chars=max_chars):
            chunk = chunk.strip()
            if not chunk:
                continue
            yield RepoChunk(
                text=f"## Source: {text_file.relative}\n{chunk}\n",
                path=text_file.relative,
            )


# Category-aware file extension allowlist. Keep the keys lowercase to
# avoid case-mismatch surprises when the user types "Python".
_CATEGORY_EXTENSIONS: dict[str, frozenset[str]] = {
    "python": frozenset({".py", ".pyi"}),
    "devops": frozenset({
        ".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf",
        ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
        ".dockerfile", ".containerfile", ".md", ".rst",
    }),
    "frontend": frozenset({
        ".html", ".htm", ".css", ".scss", ".sass", ".less",
        ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
        ".vue", ".svelte",
    }),
}


def _matches_category(relative_path: str, categories: set[str]) -> bool:
    if "all" in categories:
        return True
    p = Path(relative_path)
    ext = p.suffix.lower()
    name = p.name.lower()
    for cat in categories:
        cat_l = cat.lower()
        if cat_l == "all":
            return True
        exts = _CATEGORY_EXTENSIONS.get(cat_l)
        if exts and ext in exts:
            return True
        # Filename matches (e.g. Dockerfile, Makefile)
        if cat_l == "devops" and name in {"dockerfile", "containerfile", "makefile"}:
            return True
    return False


# ---------------------------------------------------------------------------
# Web chunking
# ---------------------------------------------------------------------------

_SCRIPT_RE = re.compile(r"<script.*?</script>", re.IGNORECASE | re.DOTALL)
_STYLE_RE = re.compile(r"<style.*?</style>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_IGNORE_TAGS = {"script", "style", "noscript", "head", "meta", "iframe"}


def _strip_html(content: str) -> str:
    """Naive HTML-to-text: drop script/style blocks, then all tags."""
    text = _SCRIPT_RE.sub(" ", content)
    text = _STYLE_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return html_lib.unescape(text)


def iter_web_chunks(
    urls: Iterable[str],
    max_chars: int = 2000,
    *,
    fetch: Optional[Callable[[str], Optional[str]]] = None,
) -> Iterator[RepoChunk]:
    """Fetch each URL and yield a ``RepoChunk`` per text chunk.

    ``fetch`` is an optional callable ``(url) -> str | None`` that
    returns raw HTML for the URL or ``None`` on failure. If not
    provided, a stdlib ``urllib``-based fetcher is used.
    """
    fetcher = fetch or _default_fetch
    for url in urls:
        try:
            content = fetcher(url)
        except Exception:
            continue
        if not content:
            continue
        text = _strip_html(content)
        if not text:
            continue
        header = f"## Source: {url}\n"
        for chunk in chunk_text(text, max_chars=max_chars):
            chunk = chunk.strip()
            if not chunk:
                continue
            yield RepoChunk(
                text=f"{header}{chunk}\n",
                path=url,
                source="web",
            )


def _default_fetch(url: str, timeout: int = 15) -> str | None:
    """Stdlib URL fetcher. Returns HTML or ``None`` on any failure."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; AI-Code-Agent/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return None
            raw = response.read()
            encoding = response.headers.get_content_charset("utf-8")
            return raw.decode(encoding, errors="ignore")
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError, OSError):
        return None


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSplit:
    """A train/validation pair produced by :func:`split_train_validation`."""

    training: list[dict]
    validation: list[dict]

    @property
    def total(self) -> int:
        return len(self.training) + len(self.validation)

    def to_dict(self) -> dict:
        return {
            "training": self.training,
            "validation": self.validation,
        }


def split_train_validation(
    records: Iterable[dict],
    val_split: float = 0.1,
    seed: int = 42,
    min_validation: int = 1,
) -> DatasetSplit:
    """Shuffle ``records`` deterministically and split into train/val.

    Args:
        records: any iterable of dicts.
        val_split: fraction (0 < val_split < 1) reserved for validation.
        seed: PRNG seed for reproducibility.
        min_validation: floor on the validation count when the dataset
            is large enough to support it; the validation set is at
            least one record whenever the total is >= 1.
    """
    if not 0 < val_split < 1:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")
    items = [r for r in records if is_valid_record(r)]
    if not items:
        return DatasetSplit(training=[], validation=[])
    rng = random.Random(seed)
    rng.shuffle(items)
    validation_count = max(min_validation, int(len(items) * val_split))
    # Never let validation consume the entire dataset.
    validation_count = min(validation_count, max(0, len(items) - 1)) if len(items) > 1 else 0
    validation = items[:validation_count]
    training = items[validation_count:]
    return DatasetSplit(training=training, validation=validation)


# ---------------------------------------------------------------------------
# High-level dataset preparation entry points
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedDataset:
    """Result of a high-level :func:`prepare_repo_dataset` call."""

    train_path: Path
    validation_path: Path
    training_count: int
    validation_count: int

    def to_dict(self) -> dict:
        return {
            "train_path": str(self.train_path),
            "validation_path": str(self.validation_path),
            "training_count": self.training_count,
            "validation_count": self.validation_count,
        }


def prepare_repo_dataset(
    output_dir: Path,
    root: Path,
    *,
    max_chars: int = 2000,
    val_split: float = 0.1,
    seed: int = 42,
    categories: set[str] | None = None,
) -> PreparedDataset:
    """Walk ``root`` and write train.jsonl + validation.jsonl under ``output_dir``."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(iter_repo_chunks(root, max_chars=max_chars, categories=categories))
    if not chunks:
        raise RuntimeError(f"No text chunks could be extracted from {root}")
    records = [c.to_dict() for c in chunks]
    split = split_train_validation(records, val_split=val_split, seed=seed)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    write_jsonl(train_path, split.training)
    write_jsonl(val_path, split.validation)
    return PreparedDataset(
        train_path=train_path,
        validation_path=val_path,
        training_count=len(split.training),
        validation_count=len(split.validation),
    )


def prepare_web_dataset(
    output_dir: Path,
    urls: Iterable[str],
    *,
    max_chars: int = 2000,
    val_split: float = 0.1,
    seed: int = 42,
    fetch: Optional[Callable[[str], Optional[str]]] = None,
) -> PreparedDataset:
    """Fetch ``urls`` and write train.jsonl + validation.jsonl under ``output_dir``."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(iter_web_chunks(urls, max_chars=max_chars, fetch=fetch))
    if not chunks:
        raise RuntimeError("No web snapshots could be fetched for dataset creation.")
    records = [c.to_dict() for c in chunks]
    split = split_train_validation(records, val_split=val_split, seed=seed)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "validation.jsonl"
    write_jsonl(train_path, split.training)
    write_jsonl(val_path, split.validation)
    return PreparedDataset(
        train_path=train_path,
        validation_path=val_path,
        training_count=len(split.training),
        validation_count=len(split.validation),
    )


__all__ = [
    "RepoChunk",
    "DatasetSplit",
    "PreparedDataset",
    "REQUIRED_KEYS",
    "read_jsonl",
    "write_jsonl",
    "is_valid_record",
    "normalize_repo_record",
    "normalize_web_record",
    "iter_repo_chunks",
    "iter_web_chunks",
    "split_train_validation",
    "prepare_repo_dataset",
    "prepare_web_dataset",
]
