import platform
import subprocess
from pathlib import Path
from typing import List

from .config import DEFAULT_MODEL, DEFAULT_MODELS, OLLAMA_HOST, OLLAMA_PORT
from .utils import find_executable, run_process


def _ollama_command() -> List[str]:
    command = find_executable("ollama")
    if not command:
        raise FileNotFoundError("Ollama CLI not found on PATH.")
    return [command]


def check_installed() -> bool:
    return find_executable("ollama") is not None


def serve(host: str = OLLAMA_HOST, port: int = OLLAMA_PORT) -> int:
    command = _ollama_command() + ["serve", "--host", host, "--port", str(port)]
    creation_flags = 0
    if platform.system() == "Windows":
        creation_flags = subprocess.CREATE_NEW_CONSOLE

    process = subprocess.Popen(command, creationflags=creation_flags)
    print(f"Started Ollama service (PID={process.pid}) on http://{host}:{port}")
    return process.pid


def pull_model(model: str = DEFAULT_MODEL, device: str = "cpu", precision: str = "int4") -> int:
    command = _ollama_command() + ["pull", model]
    returncode, stdout, stderr = run_process(command, capture_output=True)
    if returncode != 0:
        raise RuntimeError(stderr or stdout)
    print(stdout)
    return returncode


def remove_model(model: str) -> int:
    command = _ollama_command() + ["rm", model]
    returncode, stdout, stderr = run_process(command, capture_output=True)
    if returncode != 0:
        raise RuntimeError(stderr or stdout)
    print(stdout)
    return returncode


def list_models() -> str:
    command = _ollama_command() + ["list"]
    returncode, stdout, stderr = run_process(command, capture_output=True)
    if returncode != 0:
        raise RuntimeError(stderr or stdout)
    return stdout
