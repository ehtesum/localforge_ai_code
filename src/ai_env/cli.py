import argparse
import platform
import shutil
import subprocess
from pathlib import Path
from typing import List

from .config import DEFAULT_MODEL, DEFAULT_MODELS, OLLAMA_HOST, OLLAMA_PORT, ROOCODE_PORT
from .ollama import check_installed, list_models, pull_model, remove_model, serve
from .health import run_health_check
from .roocode import ensure_roocode_config
from .train import prepare_data
from .web import prepare_web_data, serve_web_ui
from .utils import find_executable


def _start_process(command: List[str]) -> None:
    print(f"Executing: {' '.join(command)}")
    try:
        subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
    except AttributeError:
        subprocess.Popen(command)


def start_services(args: argparse.Namespace) -> None:
    if not check_installed():
        raise RuntimeError("Ollama CLI is required to start services.")

    serve(host=OLLAMA_HOST, port=OLLAMA_PORT)

    roocode_executable = find_executable("roocode")
    if roocode_executable:
        _start_process([roocode_executable, "serve", "--port", str(ROOCODE_PORT)])
        print(f"Started Roo Code on port {ROOCODE_PORT}.")
    else:
        print("Roo Code CLI not found on PATH. Skip Roo Code startup.")


def _stop_named(name: str) -> bool:
    """Stop a process by base name in a cross-platform way.

    On Windows uses ``taskkill /IM <name>.exe /F``. On POSIX (Linux, macOS,
    and the Docker container this project ships in) uses ``pkill -f <name>``
    with a fallback to ``killall <name>`` if pkill is not available.
    Returns True if the command reported success, False otherwise.
    """
    system = platform.system()
    if system == "Windows":
        result = subprocess.run(
            ["taskkill", "/IM", f"{name}.exe", "/F"],
            capture_output=True, text=True,
        )
        return result.returncode == 0
    # POSIX path
    pkill = shutil.which("pkill")
    if pkill:
        result = subprocess.run(
            [pkill, "-f", name], capture_output=True, text=True,
        )
        # pkill returns 1 when no process matched, which is the same as
        # "taskkill found nothing" - treat both as "not found".
        return result.returncode == 0
    killall = shutil.which("killall")
    if killall:
        result = subprocess.run(
            [killall, name], capture_output=True, text=True,
        )
        return result.returncode == 0
    print(f"[stop] No pkill/killall on PATH; cannot stop '{name}'.")
    return False


def stop_services(args: argparse.Namespace) -> None:
    if _stop_named("ollama"):
        print("Stopped Ollama.")
    else:
        print("Ollama process not found or could not be stopped.")

    if _stop_named("roocode"):
        print("Stopped Roo Code.")
    else:
        print("Roo Code process not found or could not be stopped.")


def health_check(args: argparse.Namespace) -> None:
    results = run_health_check()
    for service, outcome in results.items():
        status = "OK" if outcome.get("ok") == "true" else "FAIL"
        message = outcome.get("status") or outcome.get("error")
        print(f"[{status}] {service}: {message}")


def check_prereqs(args: argparse.Namespace) -> None:
    print("Checking command availability...")
    for name in ["python", "ollama", "roocode", "code"]:
        path = find_executable(name)
        print(f" - {name}: {path or 'not found'}")


def model_command(args: argparse.Namespace) -> None:
    if args.subcommand == "list":
        print(list_models())
        return
    if args.subcommand == "pull":
        pull_model(args.model, device=args.device, precision=args.precision)
        return
    if args.subcommand == "remove":
        remove_model(args.model)
        return


def config_command(args: argparse.Namespace) -> None:
    path = ensure_roocode_config()
    print(f"Roo Code config written to {path}")


def train_command(args: argparse.Namespace) -> None:
    if args.subcommand == "prepare-data":
        prepare_data(
            output_dir=Path(args.output_dir),
            max_chars=args.max_chars,
            val_split=args.validation_split,
            seed=args.seed,
        )
    elif args.subcommand == "prepare-web":
        prepare_web_data(
            output_dir=Path(args.output_dir),
            urls=args.urls,
            max_chars=args.max_chars,
            val_split=args.validation_split,
            seed=args.seed,
        )
    else:
        raise RuntimeError(f"Unknown train subcommand: {args.subcommand}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local AI development environment manager."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("start", help="Start Ollama and Roo Code services.")
    subparsers.add_parser("stop", help="Stop Ollama and Roo Code services.")
    subparsers.add_parser("health", help="Run health checks against Ollama and Roo Code.")
    subparsers.add_parser("check", help="Check prerequisites for the environment.")

    model_parser = subparsers.add_parser("model", help="Manage Ollama models.")
    model_subparsers = model_parser.add_subparsers(dest="subcommand", required=True)
    model_subparsers.add_parser("list", help="List Ollama models.")

    pull_parser = model_subparsers.add_parser("pull", help="Pull an Ollama model.")
    pull_parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    pull_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    pull_parser.add_argument("--precision", default="int4", choices=["int4", "fp16", "fp32"])

    remove_parser = model_subparsers.add_parser("remove", help="Remove an Ollama model.")
    remove_parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)

    train_parser = subparsers.add_parser("train", help="Prepare repository data for fine-tuning.")
    train_subparsers = train_parser.add_subparsers(dest="subcommand", required=True)

    prepare_parser = train_subparsers.add_parser(
        "prepare-data",
        help="Collect repository files and create JSONL chunks for fine-tuning.",
    )
    prepare_parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "training_data"),
        help="Directory to write generated training files.",
    )
    prepare_parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum characters per chunk.",
    )
    prepare_parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of chunks to reserve for validation.",
    )
    prepare_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling.",
    )

    web_parser = train_subparsers.add_parser(
        "prepare-web",
        help="Fetch public web pages and create JSONL chunks for fine-tuning.",
    )
    web_parser.add_argument(
        "--urls",
        nargs="+",
        required=True,
        help="Public URLs to snapshot into training data.",
    )
    web_parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "training_data"),
        help="Directory to write generated training files.",
    )
    web_parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum characters per chunk.",
    )
    web_parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of chunks to reserve for validation.",
    )
    web_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling.",
    )

    subparsers.add_parser("config", help="Generate Roo Code configuration.")

    web_ui_parser = subparsers.add_parser(
        "serve-web",
        help="Launch local web UI for the AI coding agent.",
    )
    web_ui_parser.add_argument("--host", default="127.0.0.1", help="Host for the web UI.")
    web_ui_parser.add_argument("--port", type=int, default=7861, help="Port for the web UI.")
    web_ui_parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Workspace root to browse and edit.",
    )

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "start":
        start_services(args)
    elif args.command == "stop":
        stop_services(args)
    elif args.command == "health":
        health_check(args)
    elif args.command == "check":
        check_prereqs(args)
    elif args.command == "model":
        model_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "config":
        config_command(args)
    elif args.command == "serve-web":
        serve_web_ui(host=args.host, port=args.port, root_dir=args.root)
    else:
        parser.print_help()
