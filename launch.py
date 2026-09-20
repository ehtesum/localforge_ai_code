"""Launch the local AI Coding Agent Web UI.

Usage:
    python launch.py                  # use defaults
    python launch.py --port 9000      # custom port
    python launch.py --no-browser     # don't auto-open the browser
    python launch.py --root <dir>     # workspace root
    python launch.py --backend ollama # use Ollama HTTP API
    python launch.py --backend local  # load transformers locally
    python launch.py --backend auto   # pick automatically (default)

Environment variables (all optional):
    OLLAMA_HOST      default: 127.0.0.1
    OLLAMA_PORT      default: 11434
    OLLAMA_MODEL     default: qwen2.5-coder:7b
    WEB_HOST         default: 127.0.0.1
    WEB_PORT         default: 7861
    WORKSPACE_ROOT   default: current directory
    AI_ENV_BACKEND   default: auto
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if not src_path.exists():
        raise SystemExit(f"Could not find src/ directory at {src_path}")
    sys.path.insert(0, str(src_path))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the local AI Coding Agent Web UI in your browser.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("WEB_HOST", "127.0.0.1"),
        help="Host to bind to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("WEB_PORT", "7861")),
        help="Port to bind to (default: 7861).",
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("WORKSPACE_ROOT", str(Path.cwd())),
        help="Workspace root directory to browse and edit (default: current dir).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open the browser after the server starts.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ollama", "local"],
        default=os.environ.get("AI_ENV_BACKEND", "auto"),
        help="Inference backend. 'ollama' uses the Ollama HTTP API; "
             "'local' loads transformers in-process. 'auto' prefers ollama when reachable.",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "127.0.0.1"),
        help="Ollama host (default: 127.0.0.1, or $OLLAMA_HOST).",
    )
    parser.add_argument(
        "--ollama-port",
        type=int,
        default=int(os.environ.get("OLLAMA_PORT", "11434")),
        help="Ollama port (default: 11434, or $OLLAMA_PORT).",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b"),
        help="Default Ollama model to advertise in the Web UI.",
    )
    parser.add_argument(
        "--wait-for-ollama",
        type=int,
        default=0,
        help="If > 0, block up to this many seconds for Ollama to become reachable.",
    )
    return parser


def _ollama_reachable(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_ollama(host: str, port: int, timeout: int) -> bool:
    print(f"[launch] Waiting up to {timeout}s for Ollama at {host}:{port}...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _ollama_reachable(host, port):
            try:
                with urllib.request.urlopen(
                    f"http://{host}:{port}/api/version", timeout=2
                ) as r:
                    if r.status == 200:
                        print("[launch] Ollama is ready.", flush=True)
                        return True
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
        time.sleep(1.0)
    return False


def _resolve_backend(args: argparse.Namespace) -> str:
    if args.backend != "auto":
        return args.backend
    if _ollama_reachable(args.ollama_host, args.ollama_port):
        print(f"[launch] Auto-detected Ollama at {args.ollama_host}:{args.ollama_port}.")
        return "ollama"
    print("[launch] Ollama not reachable; falling back to local transformers.")
    return "local"


def _build_agent(backend: str, args: argparse.Namespace):
    if backend == "ollama":
        from ai_env.ollama_agent import OllamaCodingAgent
        return OllamaCodingAgent(
            host=args.ollama_host,
            port=args.ollama_port,
            default_model=args.ollama_model,
        )
    from ai_env.agent import CodingAgent
    return CodingAgent()


def _print_banner(args: argparse.Namespace) -> None:
    print("=" * 64)
    print("  AI Coding Agent - Web UI")
    print("=" * 64)
    print(f"  Workspace : {Path(args.root).resolve()}")
    print(f"  Backend   : {args.backend}")
    print(f"  Ollama    : {args.ollama_host}:{args.ollama_port}  (model: {args.ollama_model})")
    print(f"  URL       : http://{args.host}:{args.port}")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 64)


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    _ensure_src_on_path()

    backend = _resolve_backend(args)

    if args.wait_for_ollama > 0 and backend in ("auto", "ollama"):
        if not _wait_for_ollama(args.ollama_host, args.ollama_port, args.wait_for_ollama):
            print(
                f"[launch] WARNING: Ollama not reachable after {args.wait_for_ollama}s.",
                flush=True,
            )

    _print_banner(args)

    if not args.no_browser:
        try:
            webbrowser.open(f"http://{args.host}:{args.port}")
        except Exception as exc:  # pragma: no cover
            print(f"[launch] Could not open browser automatically: {exc}")

    from ai_env.web import serve_web_ui

    agent = _build_agent(backend, args)

    try:
        serve_web_ui(host=args.host, port=args.port, root_dir=args.root, agent=agent)
    except KeyboardInterrupt:
        print("\n[launch] Shutting down...")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
