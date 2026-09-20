from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_env.cli import stop_services


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rollback the local Ollama and Roo Code environment.")
    parser.add_argument("model", nargs="?", default="qwen2.5-coder:7b", help="Model to uninstall from Ollama.")
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    print("Stopping Ollama and Roo Code services...")
    stop_services(None)

    if args.model:
        ollama_path = shutil.which("ollama")
        if not ollama_path:
            raise SystemExit("Ollama CLI not found on PATH. Install Ollama before running rollback.")

        print(f"Removing model {args.model} from Ollama cache...")
        result = subprocess.run([ollama_path, "uninstall", args.model], capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(result.stderr or result.stdout)
        print(result.stdout.strip())

    print("Rollback complete. Use 'python scripts/start_services.py' to restart services if needed.")


if __name__ == "__main__":
    main()
