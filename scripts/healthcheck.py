from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_env.cli import health_check


def main() -> None:
    health_check(None)


if __name__ == "__main__":
    main()
