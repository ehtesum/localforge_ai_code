"""Web UI for the AI coding agent - thin wrapper around the new modules.

After the modular refactor, this file is intentionally short. It is
responsible for:

  * the public ``serve_web_ui`` entry point (used by ``launch.py``
    and the ``ai-env-web`` console script),
  * the ``prepare_web_data`` function used by the
    ``ai-env train prepare-web`` CLI subcommand and
    ``scripts/prepare_web_data.py``,
  * a small CLI for ``python -m ai_env.web`` that calls
    ``prepare_web_data`` from the command line.

All HTTP, file-system, git, and data-prep logic now lives in:

  * :mod:`ai_env.server`   - ``AIWebHandler`` + bootstrap helpers
  * :mod:`ai_env.workspace` - safe paths, text file discovery, git
  * :mod:`ai_env.data`     - JSONL I/O, repo/web chunking, splits

For backwards compatibility the file also re-exports
``write_jsonl`` / ``read_jsonl`` / ``chunk_text`` / ``AIWebHandler``
so that any existing import site (``scripts/*.py``, ``tests/*.py``)
keeps working without modification.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from .agent import CodingAgent
from .config import ROOT
from .data import (
    iter_web_chunks,
    prepare_web_dataset,
    read_jsonl as _read_jsonl,
    write_jsonl as _write_jsonl,
)
from .server import AIWebHandler, build_handler, run_server
from .workspace import chunk_text as _chunk_text, write_workspace_file

# Backwards-compat re-exports. Anything that previously did
# `from ai_env.web import write_jsonl` (e.g. scripts/train_model.py)
# keeps working unchanged.
write_jsonl = _write_jsonl
read_jsonl = _read_jsonl
chunk_text = _chunk_text


__all__ = [
    "AIWebHandler",
    "build_handler",
    "chunk_text",
    "iter_web_chunks",
    "prepare_web_data",
    "read_jsonl",
    "run_server",
    "serve_web_ui",
    "write_jsonl",
    "write_workspace_file",
]


# ---------------------------------------------------------------------------
# CLI / dataset preparation
# ---------------------------------------------------------------------------


def prepare_web_data(
    output_dir: Path = ROOT / "training_data",
    urls: Optional[List[str]] = None,
    max_chars: int = 2000,
    val_split: float = 0.1,
    seed: int = 42,
) -> None:
    """Prepare a web-snapshot fine-tuning dataset.

    Thin wrapper around :func:`ai_env.data.prepare_web_dataset` that
    adds the user-facing print summary. Kept in this module (rather
    than moved to ``data.py``) because it is the CLI-side of the
    ``ai_env.web`` package and is invoked by ``ai-env train prepare-web``.
    """
    if not urls:
        raise RuntimeError("At least one URL is required for web dataset preparation.")

    result = prepare_web_dataset(
        output_dir=output_dir,
        urls=urls,
        max_chars=max_chars,
        val_split=val_split,
        seed=seed,
    )

    training_records = list(_read_jsonl(result.train_path))
    validation_records = list(_read_jsonl(result.validation_path))
    source_urls = {r["path"] for r in training_records + validation_records}

    print("Prepared training data from web snapshots")
    print(f" - source URLs: {len(source_urls)}")
    print(f" - total chunks: {len(training_records) + len(validation_records)}")
    print(f" - training examples: {len(training_records)}")
    print(f" - validation examples: {len(validation_records)}")
    print(f"Data written to: {result.train_path} and {result.validation_path}")


# ---------------------------------------------------------------------------
# Web UI server entry point
# ---------------------------------------------------------------------------


def serve_web_ui(
    host: str = "127.0.0.1",
    port: int = 7861,
    root_dir: str | Path | None = None,
    agent: object | None = None,
) -> None:
    """Start the AI coding agent Web UI.

    Args:
        host: interface to bind on (default ``127.0.0.1``).
        port: TCP port (default ``7861``).
        root_dir: workspace directory to browse/edit. Defaults to the
            repo root (``ai_env.config.ROOT``).
        agent: optional pre-built agent instance. When ``None``, a
            :class:`ai_env.agent.CodingAgent` is constructed. Pass an
            :class:`ai_env.ollama_agent.OllamaCodingAgent` (or any
            compatible object) to use the Ollama HTTP backend instead.
    """
    root_path = Path(root_dir or ROOT).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Workspace root not found: {root_path}")

    ui_dir = Path(__file__).resolve().parent / "web_ui"
    if not ui_dir.exists():
        raise FileNotFoundError(f"Web UI directory not found: {ui_dir}")

    if agent is None:
        agent = CodingAgent()
        agent.set_workspace_root(root_path)
        print(f"[WEB] Starting web UI with workspace root: {root_path}")
        agent.load_model()
    else:
        if hasattr(agent, "set_workspace_root"):
            agent.set_workspace_root(root_path)
        print(f"[WEB] Starting web UI with injected agent: {type(agent).__name__}")

    server = run_server(host, port, root_path, agent, ui_dir)
    print(f"[WEB] Listening at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[WEB] Shutdown requested. Exiting.")
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# `python -m ai_env.web` entry point (CLI for `prepare_web_data`)
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare web snapshot fine-tuning data.",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        required=True,
        help="Public URLs to snapshot and convert into training chunks.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "training_data"),
        help="Directory to write generated training files (default: %(default)s).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum characters per chunk (default: %(default)s).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of chunks reserved for validation (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling (default: %(default)s).",
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    prepare_web_data(
        output_dir=Path(args.output_dir),
        urls=args.urls,
        max_chars=args.max_chars,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
