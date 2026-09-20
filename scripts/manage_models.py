from __future__ import annotations
import argparse
import sys
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_env.ollama import list_models, pull_model, remove_model


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Ollama models for the local AI environment.")
    parser.add_argument("action", choices=["list", "pull", "remove", "status"], help="Model action to perform.")
    parser.add_argument("model", nargs="?", default="qwen2.5-coder:7b", help="Model name for pull or remove.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device to use when pulling a model.")
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    if args.action in {"list", "status"}:
        print(list_models())
        return

    if args.action == "pull":
        pull_model(args.model, device=args.device, precision="int4")
        return

    if args.action == "remove":
        remove_model(args.model)
        return


if __name__ == "__main__":
    main()
