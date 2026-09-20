from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_env.cli import check_prereqs


def check_command(name: str, command: str) -> bool:
    path = shutil.which(command)
    if path:
        print(f"[OK] {name} found: {path}")
        return True
    print(f"[MISSING] {name} not found.")
    return False


def main() -> None:
    print("=== Local AI Environment Bootstrap ===\n")

    requirements = [
        ("Python 3.11+", "python"),
        ("Ollama CLI", "ollama"),
        ("Roo Code CLI", "roocode"),
        ("PowerShell 7+", "pwsh"),
        ("VS Code", "code"),
    ]

    missing = False
    for name, command in requirements:
        if not check_command(name, command):
            missing = True

    if missing:
        print("\nSome prerequisites are missing. Follow these installation steps:")
        print("1. Install Python 3.11+ from https://www.python.org/downloads/windows/")
        print("2. Install Ollama from https://ollama.com/install")
        print("3. Install Roo Code using the latest distribution channel from the vendor")
        print("4. Install VS Code from https://code.visualstudio.com/")
        print("5. Ensure your NVIDIA drivers and CUDA support are installed for GTX 1650")
        raise SystemExit(1)

    print("\nAll essential CLI tools are present. Verifying environment with ai_env...")
    check_prereqs(None)

    print("\nNext steps:")
    print("- Run python scripts/start_services.py to start Ollama and Roo Code")
    print("- Use python scripts/manage_models.py pull qwen2.5-coder:7b to download the model")
    print("- Use python scripts/healthcheck.py to verify connectivity")


if __name__ == "__main__":
    main()
