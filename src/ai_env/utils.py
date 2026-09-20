import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


def find_executable(name: str) -> Optional[str]:
    path = shutil.which(name)
    return path


def run_process(command: List[str], capture_output: bool = False) -> Tuple[int, str, str]:
    process = subprocess.run(
        command,
        capture_output=capture_output,
        text=True,
    )
    stdout = process.stdout.strip() if process.stdout else ""
    stderr = process.stderr.strip() if process.stderr else ""
    return process.returncode, stdout, stderr


def write_json(path: Path, data: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
